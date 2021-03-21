clear
cpath = pwd;
%ipath = ['-I' cpath];
ipath1 = ['-I' '../src'];
ipath2 = ['-I' '../app'];
% mex -setup C
% int in C should be replaced by long long int, which is 8 bytes.
mex('-v','-R2017b','-Dint="long long int"',ipath1,ipath2,...
	'../app/app_matlab.c',...
    '../src/ops.c','../src/ops_multi_vec.c',...
    '../src/ops_orth.c','../src/ops_multi_grid.c',...
    '../src/ops_lin_sol.c','../src/ops_eig_sol_gcg.c','../src/ops_eig_sol_pas.c',...    
    '../app/app_lapack.c','../app/app_ccs.c','../app/app_pas.c')
%movefile('app_matlab.mexw64', 'gcge.mexw64','f')
%%
clear
format long
if 1
    n=200; h = 1/(n+1);
    A=sparse([1:(n-1),1:n,2:n],[2:n,1:n,1:(n-1)],[-1*ones(1,n-1),2*ones(1,n),-1*ones(1,n-1)],n,n);
    A=A/h;%full(A)
    B=sparse([1:n],[1:n],[1*ones(1,n)],n,n);
    B=B*h;%full(B)
else
    n=300;
    A=rand(n);
    A=A'*A;
    A=sparse(A); B=[];
end
nev=10; abs_tol=1e-2; rel_tol=1e-8;
nevMax=2*nev; blockSize=10; nevInit=nevMax;
numIterMax=100; gapMin=1e-2;
tic;
[eval,evec,nev] = ...
app_matlab(A,B,...
    nev,abs_tol,rel_tol,...
    nevMax,blockSize,nevInit,...
    numIterMax,gapMin);
toc
tic;meval=eigs(A,B,nev,'smallestabs');toc
diff = norm(eval(1:nev)-meval(1:nev))
