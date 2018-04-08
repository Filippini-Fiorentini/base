#include "reference.hpp"

namespace fem {

// constructor            
Reference::Reference (const std::size_t &n, const Triangle &t): nln(n), tri_k(t)
    {
    std::vector<Real> xt = tri_k.get_x();
    std::vector<Real> yt = tri_k.get_y();
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

// compute local stiffness and local load factor
Eigen::Matrix<Real, 3, 3> 
    Reference::build_Aloc (void) const
    {
    Eigen::Matrix<Real, 3, 3> Aloc;
    Aloc.setZero();
    Quadrature quadrature(3);
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
                Aloc(j,k) += detBJ * quadrature.get_Quadrature_weight(n) * ( (invBJ_t*Grad[k][n]) * (Grad[j][n] * invBJ_t));
                }
            }
        }
    return Aloc;
    }

Eigen::Matrix<Real, 3, 1>
    Reference::build_bloc (const Real_function_2d& f) const
    {
    Eigen::Matrix<Real, 3, 1> load;
    load.setZero();
    Quadrature quadrature(3);
    std::size_t Nq = quadrature.get_Nq();
    std::vector<std::vector<Real>> phi(nln);
    phi[0] = quadrature.eval(phi1);
    phi[1] = quadrature.eval(phi2);
    phi[2] = quadrature.eval(phi3);
    for (std::size_t i=0; i<nln; i++)
        for (std::size_t n=0; n<Nq; n++)
            {
            load[i][0] += detBJ*quadrature.get_Quadrature_weight(n) * phi[i][n] * f(pphys_2d[i].get_x(), pphys_2d[i].get_y());
            }
    return load;
    }           
}
