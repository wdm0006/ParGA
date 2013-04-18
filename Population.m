classdef Population < handle
    properties
        numMembers
        pop
        popfitness=[];
        mi
        ma
        rate
        kurt
        crossover_n
        ff
        selectionstring=[1 1 0];
        printing=1;
    end
    
    methods
        function obj=Population(fit,ps,mins,maxes,k,r,n)
            %constructor
            obj.mi=mins;
            obj.ma=maxes;
            obj.numMembers=ps;
            obj.rate=r;
            obj.kurt=k;
            obj.crossover_n=n;
            obj.ff=fit;
            for j=1:ps
               obj.pop{1,j}=Member(obj.ff,mins,maxes);
               obj.pop{1,j}.randomizeWeights();
               obj.popfitness(1,j)=obj.pop{1,j}.trigger;
            end
        end
        
        function obj=setPrintingOff(obj)
           obj.printing=0;
        end
        
        function obj=setSelectionString(obj,ss)
            obj.selectionstring=ss;
        end
        
        function [obj,out]=breed(obj)
            %sorting fitnesses
            [temp,ix]=sort(obj.popfitness); 
            if obj.printing==1
            fprintf('Fitness: %20.10f',obj.popfitness(ix(1)));
            fprintf('\n');
            end
            newpop=cell(1,obj.numMembers);
            %generating new population
            count=1;
            
            
            %how many types of selection are we using?
            num_selections=sum(obj.selectionstring);
            curr_entry_counter=1;
            
            %is elitism one of those types?
            if obj.selectionstring(1)==1
                for j=1:obj.numMembers/num_selections
                    if curr_entry_counter<=obj.numMembers
                        %allowing the best from previous generation to continue
                        newpop{1,curr_entry_counter}=obj.pop{1, ix(j)};
                        %mutation
                        if rand<obj.rate
                            weights=obj.pop{1,ix(j)}.getWeights;
                            index=ceil(rand*length(weights));
                            weights(index)=weights(index)*(((randn-1)/obj.kurt)+1);
                            newpop{1,curr_entry_counter}.setWeights(weights);
                            curr_entry_counter=curr_entry_counter+1;
                        end
                        count=count+1;
                    end
                end
            end
            
            
            %is routlette one of the selection types?
            if obj.selectionstring(2)==1
                for j=1:obj.numMembers/num_selections
                    if curr_entry_counter<=obj.numMembers
                    %picking the two genes to breed, should favor more fit genes strongly.
                    index=10000;
                    index2=10000;
                    while index>length(ix) || index2>length(ix) || index==0 || index2==0
                        index=round(abs(randn-1)*length(ix)/2);
                        index2=round(abs(randn-1)*length(ix)/2);
                    end
                    w1=obj.pop{1,ix(index)}.getWeights;
                    w2=obj.pop{1,ix(index2)}.getWeights;
                    
                    %making a blank citizen for breeding
                    newpop{1,curr_entry_counter}=Member(obj.ff,obj.mi,obj.ma);
                    
                    %crossover
                    if obj.crossover_n==0
                        %no crossover
                        mod=zeros(1,length(w1));
                        mod2=ones(1,length(w2));
                        for k=1:round(rand(1,length(mod)))
                            mod(k)=1;
                            mod2(k)=0;
                        end
                        weights=(abs(mod).*w1)+(mod2.*w2);
                    elseif obj.crossover_n==1
                        %single crossover
                        mod=zeros(1,length(w1));
                        mod2=ones(1,length(w2));
                        for k=1:ceil(rand(1,1)*length(mod))
                            mod(k)=1;
                            mod2(k)=0;
                        end
                        weights=(abs(mod).*w1)+(mod2.*w2);
                    else
                        %doublecrossover
                        points=ceil(rand(2,1)*length(w1));
                        firstcross=min(points);
                        secondcross=max(points);
                        
                        mod=horzcat(horzcat(zeros(1,firstcross),ones(1,secondcross-firstcross)),zeros(1,length(w1)-secondcross));
                        mod2=horzcat(horzcat(ones(1,firstcross),zeros(1,secondcross-firstcross)),ones(1,length(w1)-secondcross));
                        weights=(abs(mod).*w1)+(mod2.*w2);
                    end
                    %mutation
                    if rand<obj.rate
                        index=ceil(rand*length(weights));
                        weights(index)=weights(index)*(((randn-1)/obj.kurt)+1);
                    end
                    newpop{1,curr_entry_counter}.setWeights(weights);
                    count=count+1;
                    curr_entry_counter=curr_entry_counter+1;
                    end
                end
            end
            
            %is tournament one of the types of selection?
            if obj.selectionstring(3)==1
                for j=1:obj.numMembers/num_selections
                    if curr_entry_counter<=obj.numMembers
                    %picking the two genes to breed, should favor more fit genes strongly.
                    a1=ceil(rand*obj.numMembers);
                    a2=ceil(rand*obj.numMembers);
                    b1=ceil(rand*obj.numMembers);
                    b2=ceil(rand*obj.numMembers);
                    
                    if obj.popfitness(a1)>obj.popfitness(a2)
                        w1ix=a1;
                    else
                        w1ix=a2;
                    end
                    
                    if obj.popfitness(b1)>obj.popfitness(b2)
                        w2ix=b1;
                    else
                        w2ix=b2;
                    end
                    
                    w1=obj.pop{1,w1ix}.getWeights;
                    w2=obj.pop{1,w2ix}.getWeights;
                    
                    %making a blank citizen for breeding
                    newpop{1,curr_entry_counter}=Member(obj.ff,obj.mi,obj.ma);
                    
                    %crossover
                    if obj.crossover_n==0
                        %no crossover
                        mod=zeros(1,length(w1));
                        mod2=ones(1,length(w2));
                        for k=1:round(rand(1,length(mod)))
                            mod(k)=1;
                            mod2(k)=0;
                        end
                        weights=(abs(mod).*w1)+(mod2.*w2);
                    elseif obj.crossover_n==1
                        %single crossover
                        mod=zeros(1,length(w1));
                        mod2=ones(1,length(w2));
                        for k=1:ceil(rand(1,1)*length(mod))
                            mod(k)=1;
                            mod2(k)=0;
                        end
                        weights=(abs(mod).*w1)+(mod2.*w2);
                    else
                        %doublecrossover
                        points=ceil(rand(2,1)*length(w1));
                        firstcross=min(points);
                        secondcross=max(points);
                        
                        mod=horzcat(horzcat(zeros(1,firstcross),ones(1,secondcross-firstcross)),zeros(1,length(w1)-secondcross));
                        mod2=horzcat(horzcat(ones(1,firstcross),zeros(1,secondcross-firstcross)),ones(1,length(w1)-secondcross));
                        weights=(abs(mod).*w1)+(mod2.*w2);
                    end
                    %mutation
                    if rand<obj.rate
                        index=ceil(rand*length(weights));
                        weights(index)=weights(index)*(((randn-1)/obj.kurt)+1);
                    end
                    newpop{1,curr_entry_counter}.setWeights(weights);
                    count=count+1;
                    curr_entry_counter=curr_entry_counter+1;
                    end
                end
            end
            
            %randomly fill the remaining spots
            while curr_entry_counter<=obj.numMembers
                newpop{1,curr_entry_counter}=Member(obj.ff,obj.mi,obj.ma);
                newpop{1,curr_entry_counter}.randomizeWeights();
                curr_entry_counter=curr_entry_counter+1;
            end
            %testing the new fitnesses
            obj.pop=newpop;
            %calculating new fitnesses (in parallel)
            temppop=obj.pop;
            
            temppopfitness=obj.popfitness;
            for j=1:obj.numMembers
               temp1=temppop{1,j}.trigger;
               temppopfitness(1,j)=temp1;
            end
            obj.popfitness=temppopfitness;          
            out=temppopfitness;
            
            
            
        end
        
        function out=converganceCheck(obj)
           %returns the fitnesses of the most recent generation
           out=obj.popfitness(end,:); 
        end
        
        function out=totalfitness(obj)
            %returns fitnesses of all of the generations
            out=obj.popfitness;
        end
        
        function obj=setPop(obj,newpop)
            %sets new citizens for the population
            obj.pop=newpop;
        end
        
        function out=getPop(obj)
            %gets the cell array of citizens
            out=obj.pop;
        end
        
        
    end 
end