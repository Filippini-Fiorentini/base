#ifndef _HH_FEM_DEF
#define _HH_FEM_DEF

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <functional>

namespace fem
	{
	
	#if !defined REAL
	#define REAL double
	#endif

	typedef REAL Real;
	
	const std::size_t PTS_MESH_X = 5;
	const std::size_t PTS_MESH_Y = 4;
	
	struct function_2d_0
		{
			Real operator()(Real x, Real y)
			{
			return (Real) 0.0*x*y;
			}
		};
	
	struct function_2d_1
		{
			Real operator()(Real x, Real y)
			{
			return (Real) 1.0 + 0.0*x*y;
			}
		};
		
	typedef 	std::function<Real(Real,Real)> Real_function_2d;
	typedef 	std::function<std::pair<Real,Real>(Real,Real)> Vect_function_2d;
		
	Real_function_2d phi1 = [](Real x, Real y)->Real
	    {
	    return 1-x-y;
	    };
	    
	Vect_function_2d grad_phi1 = [](Real x, Real y)->std::pair<Real,Real>
	    {
	    return {-1., -1.+ 0*x*y};
	    };
	
	Real_function_2d phi2 = [](Real x, Real y)->Real
	    {
	    return x + 0*y;
	    };
	    
	Vect_function_2d grad_phi2 = [](Real x, Real y)->std::pair<Real,Real>
	    {
	    return {1., 0*x*y};
	    };   
	    
	Real_function_2d phi3 = [](Real x, Real y)->Real
	    {
	    return y + 0*x;
	    };
	    
	Vect_function_2d grad_phi3 = [](Real x, Real y)->std::pair<Real,Real>
	    {
	    return {0*x*y, 1.};
	    };   
	    
	typedef enum Files_enum { OUT_ERR, DBG, MESH, IN, SOL, EXACT };
	const std::size_t N_FILES = 6;
	const std::string output_path = "../output/";
	const std::string input_path = "../input_files/";
	const std::string DEF_OUT_ERR = output_path + "def_out_err.txt";
	const std::string DEF_DBG = output_path + "def_dbg.txt";
	const std::string DEF_MESH = output_path + "def_mesh.txt";
	const std::string DEF_IN = input_path + "input_data.txt";
	const std::string DEF_SOL = output_path + "def_sol.csv";
	const std::string DEF_EXACT = output_path + "def_exact.csv";
	
	}

#endif /* _HH_FEM_DEF */
