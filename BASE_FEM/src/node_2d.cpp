#include "node_2d.hpp"

namespace fem
{
Node_2d::operator Eigen::Matrix<Real,2,1>() const
	{
	return {x, y};
	}

void Node_2d::set_x(const Real& x_1)
	{
	x = x_1;
	}

void Node_2d::set_y(const Real& x_2)
	{
	y = x_2;
	}

void Node_2d::print(void) const
	{
	std::cout << "Point: " << x << " " << y;
	std::cout << " is boundary? " << (is_bd==1 ? "YES" : "NO") << std::endl;
	}
	
void Node_2d::print(std::ofstream &ofs) const 
    {
    ofs << "Point: " << x << " " << y;
	ofs << " is boundary? " << (is_bd==1 ? "YES" : "NO") << std::endl;;
    }

}
