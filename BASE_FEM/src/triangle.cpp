#include "triangle.hpp"

namespace fem
	{
	Triangle::Triangle(): vect(3) {}

	Triangle::Triangle(const Mesh_2d& mesh, std::vector<std::size_t> indexes): vect(3)
		{
		for(std::size_t i = 0; i < 3; ++i)
			vect[i] = mesh.get_node( indexes[i] );
		}

	Triangle::Triangle(const Node_2d& n1, const Node_2d& n2, const Node_2d& n3): vect{n1, n2, n3} {}
	
	std::vector<Real> Triangle::get_x() const
		{
		std::vector<Real> res(3);
		for(std::size_t i = 0; i < 3; ++i)
			res[i] = vect[i].get_x();
		return res;
		}

	std::vector<Real> Triangle::get_y() const
		{
		std::vector<Real> res(3);
		for(std::size_t i = 0; i < 3; ++i)
			res[i] = vect[i].get_x();
		return res;
		}

	void Triangle::set_coord(Real x, Real y, std::size_t i)
		{
		vect[i].set_x(x);
		vect[i].set_x(y);
		}

	}