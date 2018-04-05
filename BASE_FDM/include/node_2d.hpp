#ifndef _HH_NODE_2D
#define _HH_NODE_2D

#include "fdm_def.hpp"

namespace fdm
{
	class Node_2d
	{
	private:
		Real x = 0.0;
		Real y = 0.0;
		bool boundary = true;
		std::vector<std::size_t> neighbours;

	public:
		Node_2d(void) = default;
		Node_2d(const Real& x_1, const Real& x_2, bool condition, const std::vector<std::size_t>& new_neigh);
		~Node_2d(void) = default;
		
		Real get_x() const {return x;}
		Real get_y() const {return y;}
		bool is_boundary() const  {return boundary;}
		std::vector<std::size_t> get_neighbours() const {return neighbours;}
		void set_x(const Real& x_1);
		void set_y(const Real& x_2);
		void set_boundary(bool condition);
		void set_neighbours(const std::vector<std::size_t>& new_neigh);
		void print(void) const;
		void print(std::ofstream &) const;
	};
}
#endif
