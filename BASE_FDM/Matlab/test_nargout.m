function [out1, out2] = test_nargout(a, b
% This function can return either one or two objects
    if (nargout == 1)
        out1 = a * b;
    elseif (nargout == 2)
        out1 = a + b;
        out2 = a - b;
    end