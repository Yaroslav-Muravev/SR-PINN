clearvars; close all;

try
    run_batch_main('params.mat',11,20,'coarse'); 
catch e
    disp(getReport(e))
end
