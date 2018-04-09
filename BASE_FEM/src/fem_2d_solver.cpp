#include "fdm_2d_solver.hpp"

namespace fdm
{
Fdm_2d_solver::Fdm_2d_solver(Mesh_2d m, Real_function_2d f1, Real_function_2d g1):
	mesh(m), f(f1), g(g1), exact(function_2d_0()), have_exact(false), 
	A(mesh.get_n_nodes(),mesh.get_n_nodes()), b(mesh.get_n_nodes(),1), A_assembled(false),
	b_assembled(false) 
	{}

Fdm_2d_solver::Fdm_2d_solver(Mesh_2d m, Real_function_2d f1, Real_function_2d g1, Real_function_2d exact1):
	mesh(m), f(f1), g(g1), exact(exact1), have_exact(true), 
	A(mesh.get_n_nodes(),mesh.get_n_nodes()), b(mesh.get_n_nodes(),1), A_assembled(false),
	b_assembled(false) 
	{}

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
	std::size_t n_tria = mesh.n_triangles();
	std::size_t nln = mesh.get_nln();
	for (std::size_t j=0; j<n_tria; j++)
	    {
	    const std::vector<std::size_t> indices = mesh.get_vector_idx(j);
	    Triangle tria(mesh,indices);
	    Reference ref_el(nln,tria);
	    Eigen::Matrix<Real,3,3> Aloc = ref_el.build_Aloc();
	    Eigen::Matrix<Real,3,1> bloc = ref_el.build_bloc();
	    std::vector<Eigen::Triplet<Real>> triplets;
	    for (std::size_t k=0; k<3; k++)
	        {
	        for (std::size_t h=0; h<3; h++)
	            {
	            std::size_t idx1 = indices(k);
	            std::size_t idx2 = indices(h);
	            triplets.push_back(idx1,idx2,Aloc(k,h));
	            //...to be continued
	            }
	        } 
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
