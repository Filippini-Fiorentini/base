#ifndef _HH_MESH_2D
#define _HH_MESH_2D

#include "node_2d.hpp"

namespace fdm
{
	class Mesh_2d
	{
	private:
		Real L_x;
		Real L_y;
		std::vector<Node_2d> nodes;
		std::size_t nr_rows;
		std::size_t nr_cols;

	public:
		Mesh_2d(Real limit_x, Real limit_y);
		Mesh_2d(Real limit_x, Real limit_y, const std::vector<Node_2d>& mesh, std::size_t rows, std::size_t cols);
		~Mesh_2d(void) = default;
		
		const Node_2d& get_node(std::size_t i) const {return nodes[i];}
		
		std::size_t get_n_nodes (void) const {return nodes.size();}
		
		Real get_Lx (void) const {return L_x;}
		Real get_Ly (void) const {return L_y;}
		
		void print (void) const;
		void print (std::ofstream &) const;
	};
}
#endif
