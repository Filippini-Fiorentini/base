#include "node_2d.hpp"

namespace fdm
{
Node_2d::Node_2d(const Real& x_1, const Real& x_2, bool condition, const std::vector<std::size_t>& new_neigh):
	x(x_1), y(x_2), boundary(condition), neighbours(new_neigh) {}

void Node_2d::set_x(const Real& x_1)
	{
	x = x_1;
	}

void Node_2d::set_y(const Real& x_2)
	{
	y = x_2;
	}

void Node_2d::set_boundary(bool condition)
	{
	boundary = condition;
	}

void Node_2d::set_neighbours(const std::vector<std::size_t>& new_neigh)
	{
	neighbours = new_neigh;
	}

void Node_2d::print(void) const
	{
	std::cout << "Point: " << x << " " << y << "\n On the boundary? " << boundary << "\n neighbours: ";
	for(auto i: neighbours)
		std::cout << i << " ";
	std::cout << std::endl;
	}
	
void Node_2d::print(std::ofstream &ofs) const 
    {
    ofs << "Point: " << x << " " << y << "\n On the boundary? " << boundary << "\n neighbours: ";
	for(auto i: neighbours)
		ofs << i << " ";
	ofs << std::endl;
    }

}
