#ifndef _HH_FDM_2D_SOLVER
#define _HH_FDM_2D_SOLVER 

#include "mesh_2d.hpp"
#include <functional>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU>

namespace fdm
{
class Fdm_2d_solver
	{
	using Real_function_2d = std::function<Real(Real,Real)>;
	
	private:
		Mesh_2d mesh;
		Real_function_2d f;
		Real_function_2d g;
		Real_function_2d exact;
		bool have_exact;

		Eigen::SparseMatrix<Real> A;
		Eigen::Matrix<Real, Eigen::Dynamic, 1> b;
		bool A_assembled;
		bool b_assembled;

	public:
		Fdm_2d_solver(Mesh_2d m, Real_function_2d f1, Real_function_2d g1);
		Fdm_2d_solver(Mesh_2d m, Real_function_2d f1, Real_function_2d g1, 
		         Real_function_2d exact1);
		~Fdm_2d_solver(void) = default;
		
		void set_f(Real_function_2d f1);
		void set_g(Real_function_2d g1);
		void set_exact(Real_function_2d g1);
		void assembly(void);
		void load_factor(void);
		Eigen::Matrix<Real, Eigen::Dynamic, 1> solve(void);
		Eigen::Matrix<Real, Eigen::Dynamic, 1> exact_sol(void); 
	};
}
#endif
