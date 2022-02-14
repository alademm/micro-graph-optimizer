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

#pragma once

#include <stdio.h>
#include <vector>
#include <array>
#include <Eigen/Core>

#ifdef MGO_ENABLE_LOG
#define MGO_LOG(...) \
    printf("%s:%i: ", __func__, __LINE__); \
    printf(__VA_ARGS__); \
    printf("\n");
#else
#define MGO_LOG(...) ((void)0)
#endif // MGO_ENABLE_LOG

#ifdef MGO_ENABLE_ASSERT
#define MGO_ASSERT(expr) \
    if (!!(expr)); \
    else { \
    fprintf(stderr, "Assertion failed: %s at %s:%i (%s)\n", #expr, __FILE__, __LINE__, __func__); \
    std::abort(); }
#else
#define MGO_ASSERT(expr) ((void)0)
#endif // MGO_ENABLE_ASSERT


namespace mgo
{
  class Variable
  {
  public:
    // To read more about the following manifold related methods (dim, plus, minus) refer to: 
    // C. Hertzberg, R. Wagner, U. Frese and L. Schroder, Integrating Generic Sensor Fusion 
    // Algorithms with Sound State Representations through Encapsulation of Manifolds.

    // Return the dimensionality of the tangent space. This is the same dimensionality of the
    // vector delta passed to plus() and the vector returned from minus().
    virtual int dim()const = 0;

    // Implements retraction operation (box-plus operator). This updates the current state of 
    // the variable (which lives on the manifold) by the small vector delta expressed in the 
    // local tangent space. If the state lives in Euclidean space then this will simply be vector addition. 
    virtual void plus(const Eigen::VectorXd& delta) = 0;

    // Returns the difference between the two states.
    virtual Eigen::VectorXd minus(const Variable& other)const = 0;

    // Whether this variable is fixed in optimization.
    bool fixed = false;
  };

  class Factor
  {
  public:
    static constexpr int kMaxVariables = 2;
    int num_variables()const { return m_num_variables; }

    void add_variable(Variable* v)
    {
      MGO_ASSERT(m_num_variables < kMaxVariables);
      m_variables[m_num_variables] = v;
      ++m_num_variables;
    }

    Variable* variable_at(int idx)const
    {
      MGO_ASSERT(m_num_variables > 0 && idx < m_num_variables);
      return m_variables[idx];
    }

    // Dimensionality of the error.
    virtual int dim()const = 0;

    virtual Eigen::VectorXd error()const = 0;

    // Jacobian wrt to the variable at idx. Defaults 
    // to computing the jacobian numerically.
    virtual Eigen::MatrixXd jacobian(int idx)const;

  private:
    Eigen::MatrixXd compute_numerical_jacobian(Variable*)const;
    std::array<Variable*, kMaxVariables> m_variables;
    int m_num_variables = 0;
  };

  class FactorGraph
  {
  public:
    FactorGraph();
    ~FactorGraph();

    // Factors and variables will be deleted upon graph destruction.
    void add_factor(Factor* f) { m_factors.push_back(f); };
    void add_variable(Variable* v) { m_variables.push_back(v); };

    std::vector<Factor*>& get_factors() { return m_factors; }
    const std::vector<Factor*>& get_factors()const { return m_factors; }
    std::vector<Variable*>& get_variables() { return m_variables; }
    const std::vector<Variable*>& get_variables()const { return m_variables; }

  private:
    std::vector<Factor*> m_factors;
    std::vector<Variable*> m_variables;
  };

  struct OptimizationParameters
  {
    int max_iterations = 100; // Maximum number of iterations.
    double relative_error_th = 1e-5; // Maximum relative error decrease.
    double absolute_error_th = 1e-5; // Maximum absolute error decrease.
  };

  // Optimize the graph using Gauss-Newton method. Returns true on convergence.
  bool optimize_gn(FactorGraph* graph, const OptimizationParameters& params = {});
}
