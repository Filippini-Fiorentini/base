#include "mesh_2d.hpp"

namespace fdm
{
Mesh_2d::Mesh_2d(Real limit_x, Real limit_y): L_x(limit_x), L_y(limit_y), nodes(PTS_MESH*PTS_MESH), nr_rows(PTS_MESH), nr_cols(PTS_MESH)
{
Real little_h_x = L_x * RELATIVE_H;
Real big_h_x = little_h_x * PRECISION_RAPPORT;
Real little_h_y = L_y * RELATIVE_H;
Real big_h_y = little_h_y * PRECISION_RAPPORT;

for(std::size_t i = 0; i < PTS_MESH*PTS_MESH; ++i)
	{
	std::size_t row_idx = i / PTS_MESH;
	std::size_t col_idx = i % PTS_MESH;
	Real x_position = 0.0;
	Real y_position = 0.0;
	if( row_idx < PTS_FINE_MESH)
		y_position = little_h_y*row_idx;
	else
		{
		if( row_idx < PTS_FINE_MESH + PTS_BIG_MESH)
			y_position = little_h_y * PTS_FINE_MESH + big_h_y * (row_idx -
			PTS_FINE_MESH);
		else
			y_position = little_h_y * PTS_FINE_MESH + big_h_y * PTS_BIG_MESH +
			 (row_idx - PTS_FINE_MESH - PTS_BIG_MESH) * little_h_y;
		}
	
	if( col_idx < PTS_FINE_MESH)
		x_position = little_h_x*col_idx;
	else
		{
		if( col_idx < PTS_FINE_MESH + PTS_BIG_MESH)
			x_position = little_h_x * PTS_FINE_MESH + big_h_x * (col_idx -
			PTS_FINE_MESH);
		else
			x_position = little_h_x * PTS_FINE_MESH + big_h_x * PTS_BIG_MESH +
			(col_idx - PTS_FINE_MESH - PTS_BIG_MESH) * little_h_x;
		}

	nodes[i].set_x(x_position);
	nodes[i].set_y(y_position);
	if(row_idx == 0 || row_idx == PTS_MESH-1 || col_idx == 0 || 
	col_idx == PTS_MESH-1)
		nodes[i].set_boundary(true);
	else
		{
		nodes[i].set_boundary(false);
		std::vector<std::size_t> new_neigh(4);
		new_neigh[0] = i - PTS_MESH;
		new_neigh[1] = i + PTS_MESH;
		new_neigh[2] = i - 1;
		new_neigh[3] = i + 1;
		nodes[i].set_neighbours(new_neigh); 
		}
	}
}

Mesh_2d::Mesh_2d(Real limit_x, Real limit_y, const std::vector<Node_2d>& mesh, std::size_t rows, std::size_t cols):
	L_x(limit_x), L_y(limit_y), nodes(mesh), nr_rows(rows), nr_cols(cols) {}

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
