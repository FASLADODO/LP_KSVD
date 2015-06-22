function [ out ] = kernel( a,b )
mu=0.5;
out=exp(-mu*(norm((a-b),2))^2);

end

