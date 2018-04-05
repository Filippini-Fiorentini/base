#include "reference.hpp"

namespace fem {

// constructor with all parameters (except for matrices)
Reference::Reference (const Real_function_2d &f1, const Real_function_2d &f2, 
                      const Real_function_2d &f3, const Vect_function_2d &gf1,
                      const Vect_function_2d &gf2, const Vect_function_2d &gf3,
                      const std::size_t &n, const Triangle &t):
    Phi1(f1), Phi2(f2), Phi3(f3), GradPhi1(gf1), GradPhi2(gf2), GradPhi3(gf3), nln(n)
    {
    std::vector<Real> xt = t.get_x();
    std::vector<Real> yt = t.get_y();
    BJ(0,0) = xt[1] - xt[0];
    BJ(0,1) = xt[2] - xt[0];
    BJ(1,0) = yt[1] - yt[0];
    BJ(1,1) = yt[2] - yt[0];
    invBJ_t = BJ.inverse().transpose();
    Eigen::Matrix<Real,2,1> c;
    c << xt[0] << yt[0];
    std::vector<Eigen::Matrix<Real,1,2>> vertices(3);
    vertices[0] << xt[0], yt[0];
    vertices[1] << xt[1], yt[1];
    vertices[2] << xt[2], yt[2];
    for (std::size_t j=0; j<3; j++)
        {
        Eigen::Matrix<Real,2,1> pp = BJ * vertices[j] + c;
        pphys_2d.set_coord(pp(0,0),pp(1,0),j);
        } 
    }

// constructor            
Reference::Reference (const std::size_t &n, const Triangle &t): nln(n)
    {
    
    }
    
// compute local stiffness and local load factor
void
Reference::local_assembly (const Quadrature &quadrature)
    {
    std::size_t Nq = quadrature.get_Nq();
    std::vector<std::vector<Node_2d>> Grad(nln);
    Grad[0] = quadrature.eval(GradPhi1);
    Grad[1] = quadrature.eval(GradPhi2);
    Grad[2] = quadrature.eval(GradPhi3);
    for (std::size_t j=0; j<nln; j++)
        {
        for (std::size_t k=0; k<nln; k++)
            {
            for (std::size_t n=0; n<Nq; n++)
                {
                Aloc(j,k) += 
                }
            }
        }
    }

}
