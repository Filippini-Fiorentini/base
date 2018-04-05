#ifndef _HH_TRIANGLE
#define _HH_TRIANGLE 

#include "mesh_2d.hpp"

namespace fem
	{
	class Triangle
		{
		private:
			std::vector<Node_2d> vect;
		public:
			Triangle();
			Triangle(const Mesh_2d& mesh, std::vector<std::size_t> indexes);
			Triangle(const Node_2d& n1, const Node_2d& n2, const Node_2d& n3);
			~Triangle() = default;

			std::vector<Real> get_x() const;
			std::vector<Real> get_y() const;

			set_coord(Real x, Real y, std::size_t i);
		};
	}

#endif