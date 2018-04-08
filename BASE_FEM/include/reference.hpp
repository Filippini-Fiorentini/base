#ifndef _HH_REFERENCE
#define _HH_REFERENCE

#include "quadrature.hpp"

namespace fem {

    class Reference {
    
        private:
            // number of local nodes
            std::size_t nln = 0;
            // Real triangle
            Triangle tri_k;
            //    
		    Eigen::Matrix<Real,2,2> BJ;
		    //    
		    Eigen::Matrix<Real,2,2> invBJ_t;
		    //    
		    Real detBJ;
		    //
		    Triangle pphys_2d;
        
        public:
        
            // constructor with no parameters
            Reference (void) = default;
            // constructor            
            Reference (const std::size_t &n, const Triangle &t);            
            
            // destructor
            ~Reference (void) = default;
            
            // get local stiffness
            Eigen::Matrix<Real, 3, 3>
            build_Aloc (void) const;
            
            //Eigen::Matrix<Real, 3, 3> build_Adv1_loc (void) const;
            
            //Eigen::Matrix<Real, 3, 3> build_Adv2_loc (void) const;
            
            //Eigen::Matrix<Real, 3, 3> build_Mloc (void) const;
            
            // get local load factor
            Eigen::Matrix<Real, 3, 1>
            build_bloc (const Real_function_2d& f) const;
            
            //
            Triangle
            get_pphys_2d (void) const {return pphys_2d;}
    
    };

}

#endif /* _HH_REFERENCE */
