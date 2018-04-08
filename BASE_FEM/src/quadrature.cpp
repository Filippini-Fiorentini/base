#include "quadrature.h"

namespace fem {

Quadrature::Quadrature(const std::size_t &n):
    nodes(n), weights(n), Nq(n)
    {
    if (Nq == 3)
        {
        weights[0] = 1/3;
        weights[1] = 1/3;
        weights[2] = 1/3;
        Node_2d nd1(0.5,0.0);
        Node_2d nd2(0.5,0.5);
        Node_2d nd3(0.0,0.5);
        nodes[0] = nd1;
        nodes[1] = nd2;
        nodes[2] = nd3;
        }
    else
        {
        std::cerr << "ERROR: at the moment, only quadrature rules with 3 nodes are supported"
                  << std::endl;        
        std::exit(1); // IDEA in this way the code stops immediately
        }
    }
        
Quadrature::Quadrature(const std::vector<Node_2d> &nd, const std::vector<Real> &w):
    nodes(nd), weights(w), Nq(nodes.size())
    {}    
    
std::vector<Node_2d> 
Quadrature::eval (const Vect_function_2d &f) const
    {
    std::vector<Node_2d> values(Nq);
    for (std::size_t j=0; j<Nq; j++)
        values[j] = f(nodes[j].get_x(), nodes[j].get_y());
    return values;
    }   

std::vector<Real> 
Quadrature::eval (const Real_function_2d &f) const
    {
    std::vector<Real> values(Nq);
    for (std::size_t j=0; j<Nq; j++)
        values[j] = f( nodes[j].get_x(), nodes[j].get_y() );
    return values;
    }
}
