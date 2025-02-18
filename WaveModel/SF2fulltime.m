function hz_slowflow_full=SF2fulltime(t_full,ts,slow_vars,ws,N_fmodes)
% Construction sea surface elevation from the slowly varying amplitudes.

% Initialize
hz_slowflow_full=zeros(1,length(t_full));
for tt=1:length(t_full)
    
    % Slowly varying mean (u_0 in equation (1))
    
    hz_slowflow_full(tt)=interp1(ts,slow_vars(1,:),t_full(tt),'linear','extrap');
    for iter_fmodes=1:N_fmodes
        % Equation (1)
        hz_slowflow_full(tt)=hz_slowflow_full(tt)+interp1(ts,slow_vars(1+iter_fmodes,:),t_full(tt),'linear','extrap')*cos(ws(iter_fmodes)*t_full(tt))...
            +  interp1(ts,slow_vars(1+N_fmodes+iter_fmodes,:),t_full(tt),'linear','extrap')*sin(ws(iter_fmodes)*t_full(tt));
    end
end
end
