classdef GAGlobe < handle
    properties (SetAccess = private)
        numGenerations
        nations
        pop_size
        mi
        ma
        numPops
        ku
        ra
        cn
        ff
        printing=1;
    end
    
    methods
        function obj=GAGlobe(fit,np,numGens,ps,mins,maxes,k,r,n)
            %constructor
            obj.numGenerations=numGens;
            obj.pop_size=ps;
            obj.nations=cell(1,4);
            obj.mi=mins;
            obj.ma=maxes;
            obj.numPops=np;
            obj.ku=k;
            obj.cn=n;
            obj.ra=r;
            obj.ff=fit;
            %initializing the 4 populations
            for j=1:obj.numPops
                obj.nations{1,j}=Population(obj.ff,ps, mins, maxes,obj.ku,obj.ra,obj.cn);
            end
        end
        
        function obj=setPrintingOff(obj)
            obj.printing=0;
            for j=1:obj.numPops
                obj.nations{1,j}.setPrintingOff;
            end
        end
        
        function obj=evolveCommunities(obj)
            %initial evolution of still segregated populations
            for m=1:obj.numPops
                if obj.printing==1
                    fprintf('\nPopulation %1.0f', m);
                    fprintf('\nMost fit citizens of Generation:\n');
                end
                for j=1:obj.numGenerations
                    [obj.nations{1,m},temp]=obj.nations{1,m}.breed;
                end
            end
        end
        
        function obj=setSelectionString(obj, pop,ss)
            obj.nations{1,pop}.setSelectionString(ss);
        end
        
        function obj=mergeCommunities(obj)
            %combining the best members of each population into one population
            if obj.printing==1
            fprintf('\nCombined Population');
            fprintf('\nMost fit citizens of Generation:\n');
            end
            newpop=cell(1,obj.pop_size);
            counter=1;
            for m=1:1:obj.numPops
                fitnesses=obj.nations{1,m}.converganceCheck;
                [~,ix]=sort(fitnesses);
                for j=1:1:obj.pop_size/obj.numPops
                    if counter<=obj.pop_size
                        newpop{1,counter}=obj.nations{1,m}.getPop{1,ix(j)};
                        counter=counter+1;
                    end
                end
            end
            obj.nations{1,end+1}=Population(obj.ff,obj.pop_size, obj.mi, obj.ma,obj.ku,obj.ra,obj.cn);
            obj.nations{1,end}=obj.nations{1,end}.setPop(newpop);
            if obj.printing==0
            obj.nations{1,end}.setPrintingOff;
            end
        end
        
        function obj=evolveGlobe(obj)
            %evlove the combined population
            for j=1:obj.numGenerations
                [obj.nations{1,end},temp]=obj.nations{1,end}.breed;
            end
        end
        
        function obj=setNumGens(obj,ng)
            obj.numGenerations=ng;
        end
        
        function obj=setKurt(obj,k)
            obj.ku=k;
        end
        
        function obj=setRate(obj,r)
            obj.ra=r;
        end
        
        function obj=selectCrossover(obj,c)
            obj.cn=c;
        end
    end
    
end