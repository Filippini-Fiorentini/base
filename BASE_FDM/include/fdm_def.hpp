#ifndef _HH_FDM_DEF
#define _HH_FDM_DEF

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace fdm
	{
	#if !defined REAL
	#define REAL double
	#endif

	typedef REAL Real;

	const std::size_t PTS_FINE_MESH = 30;
	const std::size_t PTS_BIG_MESH = 40;
	const std::size_t PTS_MESH = 2*PTS_FINE_MESH + PTS_BIG_MESH;
	const Real PRECISION_RAPPORT = 5.0;	
	const Real RELATIVE_H = 1/(2*PTS_FINE_MESH +
	                PRECISION_RAPPORT*PTS_BIG_MESH);	
	
	struct function_2d_0
		{
			Real operator()(Real x, Real y)
			{
			return (Real) 0.0;
			}
		};
	
	struct function_2d_1
		{
			Real operator()(Real x, Real y)
			{
			return (Real) 1.0;
			}
		};
	typedef enum Files_enum { OUT_ERR, DBG, IN, SOL, EXACT };
	const std::size_t N_FILES = 5;
	const std::string output_path = "../output/";
	const std::string input_path = "../input_files/";
	const std::string DEF_OUT_ERR = output_path + "def_out_err.txt";
	const std::string DEF_DBG = output_path + "def_dbg.txt";
	const std::string DEF_IN = input_path + "input_data.txt";
	const std::string DEF_SOL = output_path + "def_sol.csv";
	const std::string DEF_EXACT = output_path + "def_exact.csv";
	}

#endif
