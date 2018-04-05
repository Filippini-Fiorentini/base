#ifndef _HH_REFERENCE
#define _HH_REFERENCE

#include "quadrature.hpp"

#include <Eigen/Dense>

namespace fem {

    class Reference {
    
        private:
        
            // functions phi on the reference element
            Real_function_2d Phi1 = phi1;
            Real_function_2d Phi2 = phi2;
            Real_function_2d Phi3 = phi3;
            // gradients of the functions phi on the reference element
            Vect_function_2d GradPhi1 = grad_phi1;
            Vect_function_2d GradPhi2 = grad_phi2;
            Vect_function_2d GradPhi3 = grad_phi3;
            // number of local nodes
            std::size_t nln = 0;
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
            // constructor with all parameters (except for matrices)
            Reference (const Real_function_2d &f1, const Real_function_2d &f2, 
                        const Real_function_2d &f3, const Vect_function_2d &gf1,
                        const Vect_function_2d &gf2, const Vect_function_2d &gf3,
                        const std::size_t &n, const Triangle &t);
            // constructor            
            Reference (const std::size_t &n, const Triangle &t);            
            
            // destructor
            ~Reference (void) = default;
            
            // get local stiffness
            Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>
            build_Aloc (void) const;
            
            // get local load factor
            Eigen::Matrix<Real, Eigen::Dynamic, 1>
            build_bloc (void) const;
            
            //
            Triangle
            get_pphys_2d (void) const {return pphys_2d;}
    
    };

}

#endif /* _HH_REFERENCE */
