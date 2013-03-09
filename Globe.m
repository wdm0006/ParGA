classdef Globe < handle
    properties (SetAccess = private)
        numGenerations
        nations
        pop_size
        mi
        ma
        numPops
        ku
        ra
    end
    
    methods
        function obj=Globe(np,numGens,ps,mins,maxes,k,r)
            %constructor
            obj.numGenerations=numGens;
            obj.pop_size=ps;
            obj.nations=cell(1,4);
            obj.mi=mins;
            obj.ma=maxes;
            obj.numPops=np;
            obj.ku=k;
            obj.ra=r;
            %initializing the 4 populations
            for j=1:obj.numPops
                obj.nations{1,j}=Population(ps, mins, maxes,obj.ku,obj.ra);
            end
        end
        
        function obj=evolveCommunities(obj)
            %initial evolution of still segregated populations
            for m=1:obj.numPops
                fprintf('\nPopulation %1.0f', m);
                fprintf('\nMost fit citizens of Generation:\n');
                for j=1:obj.numGenerations
                    [obj.nations{1,m},temp]=obj.nations{1,m}.breed;
                end
            end
        end
        
        function obj=mergeCommunities(obj)
            %combining the best 25% of each population into one population
            fprintf('\nCombined Population');
            fprintf('\nMost fit citizens of Generation:\n');
            newpop=cell(1,obj.pop_size);
            counter=1;
            for m=1:1:obj.numPops
               fitnesses=obj.nations{1,m}.converganceCheck;
               [~,ix]=sort(fitnesses);
               for j=1:1:obj.pop_size/obj.numPops
                    newpop{1,counter}=obj.nations{1,m}.getPop{1,ix(j)}; 
                    counter=counter+1;
               end
            end
            obj.nations=Population(obj.pop_size, obj.mi, obj.ma,obj.ku,obj.ra);
            obj.nations=obj.nations.setPop(newpop);
        end
        
        function obj=evolveGlobe(obj)
            %evlove the combined population
            for j=1:obj.numGenerations
                [obj.nations,temp]=obj.nations.breed;
            end
        end
        
        
    end
    
end