#include "fdm_2d_solver.hpp"

namespace fdm
{
Fdm_2d_solver::Fdm_2d_solver(Mesh_2d m, Real_function_2d f1, Real_function_2d g1):
	mesh(m), f(f1), g(g1), exact(function_2d_0()), have_exact(false),
		A(mesh.get_n_nodes(), mesh.get_n_nodes()), b(mesh.get_n_nodes(),1), A_assembled(false), b_assembled(false) {}

Fdm_2d_solver::Fdm_2d_solver(Mesh_2d m, Real_function_2d f1, Real_function_2d g1, Real_function_2d exact1):
	mesh(m), f(f1), g(g1), exact(exact1), have_exact(true),
		A(mesh.get_n_nodes(), mesh.get_n_nodes()), b(mesh.get_n_nodes(),1), A_assembled(false), b_assembled(false) {}

void Fdm_2d_solver::set_f(Real_function_2d f1)
	{
	f = f1;
	}

void Fdm_2d_solver::set_g(Real_function_2d g1)
	{
	g = g1;
	}

void Fdm_2d_solver::set_exact(Real_function_2d exact1)
	{
	exact = exact1;
	have_exact = true;
	}

void Fdm_2d_solver::assembly(void)
	{
	std::size_t vect_size = mesh.get_n_nodes();
		for (std::size_t i = 0; i < vect_size; ++i)
			{
			Node_2d node = mesh.get_node(i);
			if(node.is_boundary())
				A.insert(i,i) = 1;
			else
				{
				std::vector<std::size_t> neigh_idx = node.get_neighbours();
				
				Node_2d nord = mesh.get_node( neigh_idx[0] );
				Node_2d sud = mesh.get_node( neigh_idx[1] );
				Node_2d west = mesh.get_node( neigh_idx[2] );
				Node_2d est = mesh.get_node( neigh_idx[3] );
				
				double h_n = nord.get_y() - node.get_y();
				double h_s = node.get_y() - sud.get_y();
				double h_w = node.get_x() - west.get_x();
				double h_e = est.get_x() - node.get_x();
				
				double alpha_n = 2/( h_n * (h_n + h_s) );
				double alpha_s = 2/( h_s * (h_n + h_s) );
				double alpha_w = 2/( h_w * (h_e + h_w) );
				double alpha_e = 2/( h_e * (h_e + h_w) );
				double alpha = alpha_n + alpha_s + alpha_w + alpha_e;

				// HO UN DUBBIO se sia A(i,neigh_idx[0]) oppure A(neigh_idx[0],i)
				A.insert(i,i) = alpha;
				A.insert(i,neigh_idx[0]) = -alpha_n;
				A.insert(i,neigh_idx[1]) = -alpha_s;
				A.insert(i,neigh_idx[2]) = -alpha_w;
				A.insert(i,neigh_idx[3]) = -alpha_e;
				}
			}
	}

void Fdm_2d_solver::load_factor(void)
	{
	std::size_t vect_size = mesh.get_n_nodes();
	for (std::size_t i = 0; i < vect_size; ++i)
		{
		Node_2d node = mesh.get_node(i);
		if(node.is_boundary())
			b(i,0) = g(node.get_x(), node.get_y());
		else
			b(i,0) = f(node.get_x(), node.get_y());
		}
	}

Eigen::Matrix<Real, Eigen::Dynamic, 1> Fdm_2d_solver::solve(void)
	{
	if (!A_assembled)
		{
		assembly();
		A_assembled = true;
		}

	if (!b_assembled)
		{
		load_factor();
		b_assembled = true;
		}

	A.makeCompressed();	
	Eigen::SparseLU<Eigen::SparseMatrix<Real>,Eigen::COLAMDOrdering<int>> LUsolv;
	LUsolv.analyzePattern(A); 
	LUsolv.factorize(A); 
	Eigen::Matrix<Real, Eigen::Dynamic, 1> u = LUsolv.solve(b);
	return u;
	}
	
Eigen::Matrix<Real, Eigen::Dynamic, 1> Fdm_2d_solver::exact_sol(void) 
    {
    Eigen::Matrix<Real, Eigen::Dynamic, 1> es;
    es.resize(mesh.get_n_nodes(),1);
    if ( have_exact )
        {
        std::size_t vect_size = mesh.get_n_nodes();
	    for (std::size_t i = 0; i < vect_size; ++i)
		    {
		    Node_2d node = mesh.get_node(i);
			es(i,0) = exact(node.get_x(), node.get_y());
			}
		}
    else
        {
        std::cerr << "The exact solution is not available" << std::endl;
        }    
    return es;
    }
    
}
