#ifndef _HH_QUADRATURE
#define _HH_QUADRATURE

#include "triangle.hpp"
#include <cstdlib>

namespace fem {

    class Quadrature {
    
        private:
        
            std::vector<Node_2d> nodes;
            std::vector<Real> weights;
            std::size_t Nq = 0;
    
        public:
        
            Quadrature(void) = default;
            Quadrature(const std::size_t &n);
            Quadrature(const std::vector<Node_2d> &nd, const std::vector<Real> &w);
            
            std::size_t 
            get_Nq (void) const {return Nq;}
            
            Node_2d 
            get_Quadrature_node (std::size_t j) const {return nodes[j];}
            
            Real 
            get_Quadrature_weight (std::size_t j) const {return weights[j];}
            
            std::vector<Node_2d> 
            eval (const Vect_function_2d &f) const;
            
            std::vector<Real> 
            eval (const Real_function_2d &f) const;
    
    };

}

#endif /* _HH_QUADRATURE */
