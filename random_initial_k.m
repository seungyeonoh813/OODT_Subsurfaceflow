function out_data = random_initial_k(input_u, dt_control, is_reset, init_p, k_mD)
    % -----------------------------------------------------------------
    % Python RL Environment use MRST Step-by-Step Solver
    % -----------------------------------------------------------------

    % Load MRST Module
    if isempty(whos('global', 'mrst_loaded'))
         startup;
         mrstModule add ad-core ad-blackoil
         global mrst_loaded; mrst_loaded = true;
    end

    % State initialize or load
    if is_reset
        G = cartGrid([32, 32], [500, 500]); 
        G = computeGeometry(G);
        
        perm = k_mD * milli * darcy;  %k random
        rock = makeRock(G, perm, 0.2);
        
        fluid = initSimpleADIFluid('phases', 'W', ...
                                   'mu', [1 * centi * poise], ...
                                   'rho', [1014 * kilogram/meter^3], ...
                                   'c', [1e-5 / barsa]);
        
        model = WaterModel(G, rock, fluid);
        state = initResSol(G, init_p * barsa, [1]);  %intial value random
        
        assignin('base', 'stored_model', model);
        assignin('base', 'stored_state', state);
        assignin('base', 'stored_G', G);
        assignin('base', 'stored_rock', rock);
    else
        model = evalin('base', 'stored_model');
        state = evalin('base', 'stored_state');
        G     = evalin('base', 'stored_G');
        rock  = evalin('base', 'stored_rock');
        
        if isfield(state, 'wellSol')              % Make sure to delete the state from wellSol (previous simulation)
            state = rmfield(state, 'wellSol');
        end
    end
    
    % Well
    W = [];
    W = addWell(W, G, rock, 1, 'Type', 'rate', ...
                     'Val', input_u * meter^3/day, ...
                     'Name', 'Injector', 'Sign', 1, 'comp_i', [1]);
                 
    W = addWell(W, G, rock, G.cells.num, 'Type', 'bhp', ...
                     'Val', 90 * barsa, ...
                     'Name', 'Producer', 'Sign', -1, 'comp_i', [1]);
                 
    % Simulation
    schedule = simpleSchedule(dt_control * day, 'W', W);
    [ws, next_states, ~] = simulateScheduleAD(state, model, schedule);
    
    % Result (copy to the next_state, instead using state)
    next_state = next_states{1};
    
    % Since producer has m^3/sec, change it to m^3/day
    prod_rate = 0;
    if ~isempty(ws{1})
        for j = 1:numel(ws{1})
            if strcmp(ws{1}(j).name, 'Producer')
                prod_rate = ws{1}(j).qWs / (meter^3/day); 
            end
        end
    end
    
    % Store for next step
    assignin('base', 'stored_state', next_state);
    
    % Toss to python with the result (dictionary)
    out_data.pressure = next_state.pressure;
    out_data.prod_rate = prod_rate;
end