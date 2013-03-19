classdef Member < handle
    
    properties       
        gene;
        maxes;
        mins;
        fitness_fn
    end
    
    methods
        function obj=Member(ff,mi,ma)
            %constructor
            %declares the bounds for the parameters
            obj.gene=zeros(1,length(mi));
            obj.mins=mi;
            obj.maxes=ma;
            obj.fitness_fn=ff;
        end
        
        
        function out=trigger(obj)
            out=obj.fitness_fn(obj.gene);
        end
        
        function w = randomizeWeights(obj)
            %randomize the controller constants to a scale (must be positive)
            for j=1:length(obj.gene)
                obj.gene(j)=(rand*(obj.maxes(j)-obj.mins(j)))+obj.mins(j);
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