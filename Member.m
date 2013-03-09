classdef Member < handle
    
    properties       
        gene;
        maxes;
        mins;
    end
    
    methods
        function obj=Member(mi,ma)
            %constructor
            %declares the bounds for the parameters
            obj.gene=zeros(1,length(mi));
            obj.mins=mi;
            obj.maxes=ma;
        end

        
        function out=trigger(obj)
%             This section should contain any simulation, function calls 
%             or anything else included in calculating the fitness of the Objective
%             
%             out should be the fitness, and nothing else.
%             
%             note that this code is set for fitness to be MINIMIZED
            
            out=rand;

        end
        
        function w = randomizeWeights(obj)
            %randomize the controller constants to a scale (must be positive)
            for j=1:length(obj.gene)
                obj.gene(j)=rand*(obj.maxes(j)-obj.mins(j))+obj.mins(j);
            end
            w=obj.gene;
        end
        
        function obj=resetErrors(obj)
            obj.gene=zeros(1,length(obj.gene));
        end
        
        
        function w = setWeights(obj,in)
            %set constants
            obj.gene=in;
            for j=1:length(obj.gene)
                if obj.gene(j)>obj.maxes(j)
                    obj.gene(j)=obj.maxes(j);
                elseif obj.gene(j)<obj.mins(j)
                    obj.gene(j)=obj.mins(j);
                end
            end
            
            w=obj.gene;
        end
        
        function w = getWeights(obj)
            w=obj.gene;
        end
    end
end