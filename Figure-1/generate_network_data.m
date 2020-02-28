clear all; clc;
% addpath('C:\Users\brend\Documents\matpower7.0')
% startup
% define_constants
mpc = loadcase('case39')
results = rundcopf(mpc)

% results.bus(:,PD)
% results.gen(:,PG) %generation after DC OPF
% sum(results.bus(:,PD)) - sum(results.gen(:,PG))

number_buses = 39;
csvwrite('nominal_injections.csv',mpc.gen(:,[1,2]))
csvwrite('nominal_demand.csv',mpc.bus(:,3))
csvwrite('lines.csv',mpc.branch(:,[1,2]))
csvwrite('line_limits.csv',mpc.branch(:,6))
[number_lines, ~] = size(mpc.branch);
line_susceptances = NaN(number_lines,1);
for i = 1:number_lines
    if mpc.branch(i,9) == 0
       line_susceptances(i,1) =  1/mpc.branch(i,4);
    else
       line_susceptances(i,1) =  1/(mpc.branch(i,9)*mpc.branch(i,4));
    end
end
csvwrite('line_susceptances.csv',line_susceptances)
