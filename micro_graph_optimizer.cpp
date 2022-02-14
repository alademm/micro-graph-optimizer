// -----------------------------------------------------------------------------
// Copyright (c) 2022 Mohamed Aladem
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this softwareand associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright noticeand this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// -----------------------------------------------------------------------------

#include "micro_graph_optimizer.h"

#include <math.h>
#include <map>
#include <Eigen/Sparse>

namespace mgo
{
  namespace
  {
    struct VariableInfo
    {
    public:
      inline VariableInfo& set_dim(int val) { m_dim = val; return *this; }
      inline VariableInfo& set_idx(int val) { m_idx = val; return *this; }
      inline int dim()const { return m_dim; }
      inline int idx()const { return m_idx; }

    private:
      int m_dim;
      int m_idx; // The index of this variable in the H matrix.
    };

    struct ScratchPad
    {
      int total_variables_dim = 0;
      int total_factors_dim = 0;
      std::map<Variable*, VariableInfo> variable_lookup_table;
      inline const VariableInfo& variable_lookup(Variable* var) { MGO_ASSERT(variable_lookup_table.count(var) != 0); return variable_lookup_table[var]; }
      typedef Eigen::Triplet<double> T;
      std::vector<T> tripletList;
      Eigen::SparseMatrix<double> H;
      Eigen::VectorXd b;
    };

    // ===================================================================
    void compute_graph_info(const FactorGraph& graph, ScratchPad* pad)
    {
      const std::vector<Variable*>& variables = graph.get_variables();
      int total_variables_dim = 0;
      std::map<Variable*, VariableInfo> variable_lookup_table;
      for (int i = 0, var_idx = 0, count = variables.size(); i < count; ++i)
      {
        Variable* var = variables[i];
        if (!var->fixed)
        {
          const int v_dim = var->dim();
          total_variables_dim += v_dim;
          variable_lookup_table[var] = VariableInfo().set_dim(v_dim).set_idx(var_idx);
          var_idx += v_dim;
        }
      }
      pad->total_variables_dim = total_variables_dim;
      pad->variable_lookup_table.swap(variable_lookup_table);

      const std::vector<Factor*>& factors = graph.get_factors();
      int total_factors_dim = 0;
      for (int i = 0, count = factors.size(); i < count; ++i)
      {
        const int f_dim = factors[i]->dim();
        total_factors_dim += f_dim;
      }
      pad->total_factors_dim = total_factors_dim;

      pad->H = Eigen::SparseMatrix<double>(total_variables_dim, total_variables_dim);
      pad->b = Eigen::VectorXd::Zero(total_variables_dim);
    }

    // ===================================================================
    double compute_error_norm_squared(const FactorGraph& graph)
    {
      const std::vector<Factor*>& factors = graph.get_factors();
      double error = 0.0;
      for (size_t i = 0, count = factors.size(); i < count; ++i)
      {
        error += factors[i]->error().squaredNorm();
      }
      return error;
    }

    // ===================================================================
    void linearize_single_factor(Factor* factor, ScratchPad* pad)
    {
      // Our goal in this function is to add the contribution of a factor
      // to the linearized system (H, b).

      // First we need to calculate the jacobian of the factor wrt each of
      // its variables then stack them horizontally in Js then compute Jt*J matrix.
      const int n_rows = factor->dim();
      const int num_variables = factor->num_variables();
      std::vector<int> vars_cols(num_variables, -1);
      std::vector<int> vars_dim(num_variables, -1);
      int n_cols = 0;
      for (int i = 0; i < num_variables; ++i)
      {
        Variable* var = factor->variable_at(i);
        if (!var->fixed)
        {
          const int var_dim = pad->variable_lookup(var).dim();
          vars_cols[i] = n_cols;
          vars_dim[i] = var_dim;
          n_cols += var_dim;
        }
      }

      if (n_cols == 0)
      {
        MGO_LOG("All variables connected to this factor are fixed.");
        return;
      }

      Eigen::MatrixXd Js(n_rows, n_cols);
      for (int i = 0, start_col = 0; i < num_variables; ++i)
      {
        if (!factor->variable_at(i)->fixed)
        {
          Eigen::MatrixXd J = factor->jacobian(i);
          Js.block(0, start_col, J.rows(), J.cols()) = J;
          start_col += J.cols();
        }
      }

      Eigen::MatrixXd JtJ(Js.cols(), Js.cols());
      JtJ.noalias() = Js.transpose() * Js;

      // Now we need to add the contribution to H. Note that we are only filling the lower triangular part.
      for (int i = 0; i < num_variables; ++i)
      {
        if (factor->variable_at(i)->fixed)
        {
          continue;
        }
        const int H_col = pad->variable_lookup(factor->variable_at(i)).idx();
        const int JtJ_col = vars_cols[i];
        for (int j = i; j < num_variables; ++j)
        {
          if (factor->variable_at(j)->fixed)
          {
            continue;
          }
          const int H_row = pad->variable_lookup(factor->variable_at(j)).idx();
          const int JtJ_row = vars_cols[j];
          for (int JtJ_i = JtJ_col, H_i = H_col; JtJ_i < (JtJ_col + vars_dim[i]); ++JtJ_i, ++H_i)
          {
            for (int JtJ_j = JtJ_row, H_j = H_row; JtJ_j < (JtJ_row + vars_dim[j]); ++JtJ_j, ++H_j)
            {
              pad->tripletList.push_back(ScratchPad::T(H_j, H_i, JtJ(JtJ_j, JtJ_i)));
            }
          }
        }
      }

      // Handle the vector b.
      const Eigen::VectorXd Jtb = Js.transpose() * factor->error();
      for (int i = 0; i < num_variables; ++i)
      {
        Variable* var = factor->variable_at(i);
        if (!var->fixed)
        {
          const int var_idx = pad->variable_lookup(var).idx();
          const int var_dim = vars_dim[i];
          pad->b.segment(var_idx, var_dim) -= Jtb.segment(vars_cols[i], var_dim);
        }
      }
    }

    // ===================================================================
    bool iterate(FactorGraph* graph, ScratchPad* pad)
    {
      const std::vector<Factor*>& factors = graph->get_factors();
      pad->b.setZero();
      pad->tripletList.clear();
      for (size_t i = 0, count = factors.size(); i < count; ++i)
      {
        linearize_single_factor(factors[i], pad);
      }
      MGO_ASSERT(!pad->tripletList.empty());
      pad->H.setFromTriplets(pad->tripletList.begin(), pad->tripletList.end());

      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> chol(pad->H);
      Eigen::VectorXd dx = chol.solve(pad->b);
      if (chol.info() == Eigen::Success)
      {
        std::vector<Variable*>& variables = graph->get_variables();
        for (int i = 0, d = 0, count = variables.size(); i < count; ++i)
        {
          Variable* var = variables[i];
          if (!var->fixed)
          {
            const int vd = pad->variable_lookup_table[var].dim();
            var->plus(dx.segment(d, vd));
            d += vd;
          }
        }
        return true;
      }
      else
      {
        MGO_LOG("Linear solver failed.");
        return false;
      }
    }

    // ===================================================================
    bool continue_iterating_check(int iter_num, double current_error, double new_error, const OptimizationParameters& params, bool* converged)
    {
      if (!std::isfinite(new_error))
      {
        return false;
      }

      if (iter_num == params.max_iterations)
      {
        MGO_LOG("Max iterations reached.");
        return false;
      }

      const double error_decrease = current_error - new_error;
      if ((error_decrease <= params.absolute_error_th) ||
        ((error_decrease / current_error) <= params.relative_error_th))
      {
        MGO_LOG("Converged.");
        *converged = true;
        return false;
      }

      return true;
    }

    // ===================================================================
  } // anonymous namespace

  bool optimize_gn(FactorGraph* graph, const OptimizationParameters& params)
  {
    MGO_LOG("Started optimization with %i factors and %i variables.", (int)graph->get_factors().size(), (int)graph->get_variables().size());
    ScratchPad pad;
    compute_graph_info(*graph, &pad);
    double current_error = 0.5 * compute_error_norm_squared(*graph);
    MGO_LOG("Initial error: %f", current_error);
    double new_error = current_error;
    int iter_num = 0;
    bool converged = false;
    bool continue_iterating = true;
    while (continue_iterating)
    {
      current_error = new_error;
      if (!iterate(graph, &pad))
      {
        return false;
      }
      ++iter_num;
      new_error = 0.5 * compute_error_norm_squared(*graph);
      MGO_LOG("New error after iteration %i: %f", iter_num, new_error);
      continue_iterating = continue_iterating_check(iter_num, current_error, new_error, params, &converged);
    }
    return converged;
  }

  // ===================================================================
  Eigen::MatrixXd Factor::jacobian(int idx)const
  {
    MGO_ASSERT(m_num_variables > 0 && idx < m_num_variables);
    return compute_numerical_jacobian(m_variables[idx]);
  }

  // ===================================================================
  Eigen::MatrixXd Factor::compute_numerical_jacobian(Variable* v)const
  {
    constexpr double h = 1e-5;
    const int N = v->dim();
    const int M = this->dim();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(M, N);
    Eigen::VectorXd dx = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd dy0 = this->error();
    constexpr double k = 1.0 / (2.0 * h);
    for (int i = 0; i < N; ++i)
    {
      dx(i) = h;
      v->plus(dx); // right
      const Eigen::VectorXd dy1 = this->error() - dy0;
      dx(i) = -2.0 * h;
      v->plus(dx); // left
      const Eigen::VectorXd dy2 = this->error() - dy0;
      dx(i) = h;
      v->plus(dx); // return to original state.
      dx(i) = 0.0;
      J.col(i) << (dy1 - dy2) * k;
    }
    return J;
  }

  // ===================================================================
  FactorGraph::FactorGraph() = default;

  FactorGraph::~FactorGraph()
  {
    for (Factor* f : m_factors)
    {
      delete f;
    }
    for (Variable* v : m_variables)
    {
      delete v;
    }
  }
}
