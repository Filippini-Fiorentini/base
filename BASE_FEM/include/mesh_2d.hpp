#ifndef _HH_MESH_2D
#define _HH_MESH_2D

#include "node_2d.hpp"
#include <memory>

namespace fem
{
	class Mesh_2d
	{
	private:
	
		Real L_x = 0.0;
		Real L_y = 0.0;
		std::vector<Node_2d> nodes;
		std::vector<std::vector<std::size_t>> tr_indices;
		std::size_t nr_rows = 0;
		std::size_t nr_cols = 0;
		Real h_x = 0.0;
		Real h_y = 0.0;
		std::size_t degree = 1;
	    std::size_t nln = 0;

	public:
		Mesh_2d(Real limit_x, Real limit_y);
		Mesh_2d(Real limit_x, Real limit_y, const std::vector<Node_2d>& mesh, 
		                            std::size_t rows, std::size_t cols, std::size_t deg);
		~Mesh_2d(void) = default;
		
		const Node_2d& get_node(std::size_t i) const {return *nodes[i];}
		
		std::size_t get_n_nodes (void) const {return nodes.size();}
		
		Real get_Lx (void) const {return L_x;}
		Real get_Ly (void) const {return L_y;}
		
		std::size_t get_nln (void) const {return nln;}
		
		const std::vector<std::size_t> & get_vector_idx (const std::size_t &k) const
		                                        {return tr_indices[k];}
		
		void print (void) const;
		void print (std::ofstream &) const;
	};
}
#endif
