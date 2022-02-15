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

#include <iostream>

using namespace mgo;

// This basic example constructs a simple linear pose graph and optimizes it.
// Our problem here is linear and the state lives in the Euclidean space therefore
// traditional vector operations can be used.

class Pose : public Variable
{
public:
  Pose(const Eigen::Vector2d& pos) : m_position(pos) {}

  Eigen::Vector2d position()const { return m_position; }
  virtual int dim()const override { return 2; }
  virtual void plus(const Eigen::VectorXd& delta) override
  {
    m_position[0] += delta[0];
    m_position[1] += delta[1];
  }

private:
  Eigen::Vector2d m_position;
};

class PriorFactor : public Factor
{
public:
  PriorFactor(Pose* p, const Eigen::Vector2d& measurement) :
    m_measurement(measurement)
  {
    add_variable(p);
  }

  virtual int dim()const { return 2; }
  virtual Eigen::VectorXd error()const override
  {
    return (static_cast<Pose*>(this->variable_at(0))->position() - m_measurement);
  }

  virtual Eigen::VectorXd subtract_error(const Eigen::VectorXd& e1, const Eigen::VectorXd& e2)const override
  {
    return (e1 - e2);
  }

private:
  Eigen::Vector2d m_measurement;
};

class Pose2PoseFactor : public Factor
{
public:
  Pose2PoseFactor(Pose* p_a, Pose* p_b, const Eigen::Vector2d& measurement) :
    m_measurement(measurement)
  {
    add_variable(p_a);
    add_variable(p_b);
  }

  virtual int dim()const { return 2; }
  virtual Eigen::VectorXd error()const override
  {
    const auto p1 = static_cast<Pose*>(this->variable_at(0))->position();
    const auto p2 = static_cast<Pose*>(this->variable_at(1))->position();
    return ((p2 - p1) - m_measurement);
  }

  virtual Eigen::VectorXd subtract_error(const Eigen::VectorXd& e1, const Eigen::VectorXd& e2)const override
  {
    return (e1 - e2);
  }

private:
  Eigen::Vector2d m_measurement;
};

void print_poses(const FactorGraph& graph)
{
  const std::vector<Variable*>& variables = graph.get_variables();
  for (int i = 0, count = variables.size(); i < count; ++i)
  {
    Pose* p = static_cast<Pose*>(variables[i]); // We know our variables are of type Pose.
    std::cout << i << ": " << p->position().x() << " " << p->position().y() << std::endl;
  }
}

int main()
{
  FactorGraph graph;

  // Create 3 pose nodes in our graphs and just randomly set their initial values to 0.
  Pose* x1 = new Pose(Eigen::Vector2d(0.0, 0.0));
  Pose* x2 = new Pose(Eigen::Vector2d(0.0, 0.0));
  Pose* x3 = new Pose(Eigen::Vector2d(0.0, 0.0));
  graph.add_variable(x1);
  graph.add_variable(x2);
  graph.add_variable(x3);

  // Create a prior factor that fixes the first pose to origin. This is needed to constrain
  // the gauge freedom. Note that you can also explicitly mark the variable as fixed.
  PriorFactor* f1 = new PriorFactor(x1, Eigen::Vector2d(0.0, 0.0));
  graph.add_factor(f1);

  // Create some factors between the variables.
  Pose2PoseFactor* f12 = new Pose2PoseFactor(x1, x2, Eigen::Vector2d(1.0, 1.0));
  graph.add_factor(f12);
  Pose2PoseFactor* f23 = new Pose2PoseFactor(x2, x3, Eigen::Vector2d(1.0, 1.0));
  graph.add_factor(f23);
  Pose2PoseFactor* f13 = new Pose2PoseFactor(x1, x3, Eigen::Vector2d(1.0, 1.0));
  graph.add_factor(f13);

  std::cout << "Original poses:" << std::endl;
  print_poses(graph);
  mgo::optimize_gn(&graph);
  std::cout << "Optimized poses:" << std::endl;
  print_poses(graph);

  return 0;
}
