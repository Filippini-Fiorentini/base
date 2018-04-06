#ifndef _HH_INTERFACE_FEM_2D
#define _HH_INTERFACE_FEM_2D

#include "mesh_2d.hpp"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU>
#include "GetPot"

namespace fem
{

void 
write_solution_file(const Mesh_2d& mesh, const Eigen::Matrix<fem::Real, Eigen::Dynamic, 1>& vect, const std::string& filename);

Real 
read_value(const std::string& filename, const std::string& name, Real default_value);

/*
std::function<fdm::Real(fdm::Real, fdm::Real) 
read_2d_function(const std::string& filename, const std::string& name);
*/

std::vector<std::string>
input_list_file(const std::string& default_name, int argc, char* argv []);

}

#endif /* _HH_INTERFACE_FEM_2D */
