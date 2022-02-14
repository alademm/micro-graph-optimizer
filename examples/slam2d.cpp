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

#include <Eigen/Geometry>
#include <string>
#include <iostream>
#include <map>
#include <fstream>

using namespace mgo;

// This example demonstrates optimzing a non-linear 2D SLAM problem read from a g2o file.

double normalize_angle(double theta_rad)
{
  // Normalize the angle to the range [-pi, pi).
  constexpr double kPI = 3.14159265358979323846;
  constexpr double k2PI = 2.0 * kPI;
  return (theta_rad - k2PI * std::floor((theta_rad + kPI) / k2PI));
}

class Pose2d : public Variable
{
public:
  Pose2d(double x, double y, double yaw_rad) :
    m_x(x), m_y(y), m_yaw_rad(yaw_rad)
  {
  }

  virtual int dim()const override { return 3; }
  virtual void plus(const Eigen::VectorXd& delta) override
  {
    m_x += delta[0];
    m_y += delta[1];
    m_yaw_rad = normalize_angle(m_yaw_rad + delta[2]);
  }

  virtual Eigen::VectorXd minus(const Variable& other)const override
  {
    const Pose2d& s = static_cast<const Pose2d&>(other);
    Eigen::Vector3d res;
    res << (s.m_x - m_x), (s.m_y - m_y), normalize_angle(s.m_yaw_rad - m_yaw_rad);
    return res;
  }

  double x()const { return m_x; }
  double y()const { return m_y; }
  double yaw_rad()const { return m_yaw_rad; }

private:
  double m_x, m_y, m_yaw_rad;
};

class Constraint2d : public Factor
{
public:
  Constraint2d(Pose2d* v_a, Pose2d* v_b, double x_ab, double y_ab,
    double yaw_ab_rad, const Eigen::Matrix3d& sqrt_info) :
    m_pos_ab(x_ab, y_ab), m_yaw_ab_rad(yaw_ab_rad), m_sqrt_info(sqrt_info)
  {
    add_variable(v_a);
    add_variable(v_b);
  }

  virtual int dim()const { return 3; }

  virtual Eigen::VectorXd error()const override
  {
    MGO_ASSERT(this->num_variables() == 2);
    const Pose2d* v_a = static_cast<Pose2d*>(this->variable_at(0));
    const Pose2d* v_b = static_cast<Pose2d*>(this->variable_at(1));
    Eigen::Vector3d r;
    Eigen::Vector2d pos_ab_pred = { v_b->x() - v_a->x(), v_b->y() - v_a->y() };
    r.head<2>() = Eigen::Rotation2Dd(v_a->yaw_rad()).toRotationMatrix().transpose() * pos_ab_pred - m_pos_ab;
    r(2) = normalize_angle((v_b->yaw_rad() - v_a->yaw_rad()) - m_yaw_ab_rad);
    return r;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
  Eigen::Vector2d m_pos_ab;
  double m_yaw_ab_rad;
  Eigen::Matrix3d m_sqrt_info;
};

bool read_g2o(const std::string& filename, mgo::FactorGraph* graph)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    MGO_LOG("Failed to open file: %s", filename.c_str());
    return false;
  }

  std::string line;
  std::map<int, Pose2d*> id_to_pose;
  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string data_type;
    ss >> data_type;
    if (data_type == "VERTEX_SE2")
    {
      int id;
      double x, y, th;
      ss >> id >> x >> y >> th;
      Pose2d* p = new Pose2d(x, y, normalize_angle(th));
      graph->add_variable(p);
      id_to_pose[id] = p;
    }
    else if (data_type == "EDGE_SE2")
    {
      int id_a, id_b;
      double dx, dy, d_yaw, i11, i12, i13, i22, i23, i33;
      ss >> id_a >> id_b >> dx >> dy >> d_yaw >> i11 >> i12 >> i13 >> i22 >> i23 >> i33;
      Eigen::Matrix3d info_mtrx = (Eigen::Matrix3d() <<
        i11, i12, i13,
        i12, i22, i23,
        i13, i23, i33).finished();
      MGO_ASSERT(id_to_pose.count(id_a) != 0);
      MGO_ASSERT(id_to_pose.count(id_b) != 0);
      graph->add_factor(new Constraint2d(id_to_pose[id_a], id_to_pose[id_b], dx, dy,
        d_yaw, info_mtrx.llt().matrixL()));
    }
    else
    {
      MGO_LOG("Unhandled type: %s", data_type.c_str());
      return false;
    }
  }
  return true;
}

void dump_poses(const std::string& filename, const FactorGraph& graph)
{
  std::ofstream file(filename);
  if (!file.is_open())
  {
    MGO_LOG("Failed to open file: %s", filename.c_str());
    return;
  }

  const std::vector<Variable*>& variables = graph.get_variables();
  for (int i = 0, count = variables.size(); i < count; ++i)
  {
    Pose2d* p = static_cast<Pose2d*>(variables[i]);
    file << i << " " << p->x() << " " << p->y() << " " << p->yaw_rad() << std::endl;
  }
}

int main()
{
  FactorGraph graph;
  // You can get this dataset from: https://lucacarlone.mit.edu/datasets/
  if (!read_g2o("./input_M3500_g2o.g2o", &graph))
  {
    return -1;
  }

  // Fix the first variable.
  graph.get_variables()[0]->fixed = true;

  dump_poses("./original.txt", graph);
  mgo::optimize_gn(&graph);
  dump_poses("./optimized.txt", graph);

  return 0;
}
