#include "mesh_2d.hpp"

namespace fem
{

Mesh_2d::Mesh_2d(Real limit_x, Real limit_y): 
    L_x(limit_x), L_y(limit_y), nodes(PTS_MESH_X*PTS_MESH_Y), nr_rows(PTS_MESH_Y), 
    nr_cols(PTS_MESH_X), h_x(limit_x/PTS_MESH_X), h_y(limit_y/PTS_MESH_Y), tr_indices( 2 * (PTS_MESH_X - 1)  * (PTS_MESH_Y - 1))
    {
    nln = (degree+1)*(degree+2)/2;  
    // Building the nodes 
    for(std::size_t i = 0; i < PTS_MESH_X * PTS_MESH_Y; ++i)
        {
        std::size_t y_idx = i / PTS_MESH_X;
        std::size_t x_idx = i % PTS_MESH_X;
        nodes[i].set_x(x_idx * h_x);
        nodes[i].set_y(y_idx * h_y);
        }
    std::size_t counter_tri = 0;
    // Building the triangles 
    for(std::size_t i = 0; i < (PTS_MESH_X-1) * (PTS_MESH_Y-1); ++i)
        {
        tr_indices[counter_tri++] = {i,     i + 1,              i + PTS_MESH_X};
        tr_indices[counter_tri++] = {i + 1, i + 1 + PTS_MESH_X, i + PTS_MESH_X};
        }
    }

Mesh_2d::Mesh_2d(Real limit_x, Real limit_y, const std::vector<Node_2d>& mesh, std::size_t rows, std::size_t cols, std::size_t deg):
	L_x(limit_x), L_y(limit_y), nodes(mesh), nr_rows(rows), nr_cols(cols), h_x(limit_x/nr_cols), 
	        h_y(limit_y/nr_rows), degree(deg) 
	{
	nln = (degree+1)*(degree+2)/2;
	}

void Mesh_2d::print (void) const
	{
    std::cout << "Lx: " << L_x << std::endl;
    std::cout << "Ly: " << L_y << std::endl;
    for (const Node_2d &nd : nodes)
        nd.print();
	}

void Mesh_2d::print (std::ofstream &ofs) const
	{
    ofs << "Lx: " << L_x << std::endl;
    ofs << "Ly: " << L_y << std::endl;
    for (const Node_2d &nd : nodes)
        nd.print(ofs);
	}

}
