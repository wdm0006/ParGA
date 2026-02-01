classdef Member < handle
    
    properties
        gene;
        maxes;
        mins;
        fitness_fn;
        binary=false;
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
        
        function out=bi2real(obj,left,right)
            leftpart=0;
            rightpart=0;
            for j=length(left):-1:1
                leftpart=leftpart+(left(j)*2^(j-1));
                rightpart=rightpart+(right(j)*2^(j-1));
            end
            
            rightpart=rightpart*10^-10;
            out=leftpart+rightpart;
        end
        
        function obj=binarize(obj)
            obj.gene=zeros(length(obj.gene)*64,1);
            obj.binary=true;
        end
        
        function out=real_conv(obj)
            if obj.binary==true
                out=zeros(length(obj.gene)/64,1);
                for j=1:64:length(obj.gene)-64
                    out((j+63)/64)=obj.bi2real(obj.gene(j:j+31),obj.gene(j+32:j+63));
                end
            else
                out=obj.gene;
            end
        end
        
        function out=trigger(obj)
            if obj.binary==false
                out=obj.fitness_fn(obj.gene);
            else
                gene_real=obj.real_conv(obj.gene);
                out=obj.fitness_fn(gene_real);
            end
        end
        
        function out=real2bi(obj, realn)
            wholepart=floor(realn);
            decpart=realn-wholepart;
            strdecpart=str(decpart);
            if length(strdecpart)<=10
                decpart=decpart*10^10;
            elseif length(strdecpart)>10
                decpart=str2double(strdecpart(1:10))*10^10;
            end
            
            wp_bi=dec2bin(wholepart);
            dp_bi=dec2bin(decpart);
            
            out=zeroes(1,64);
            for j=1:32
                if strcmp(wp_bi(j),'1')
                    out(j)=1;
                else
                    out(j)=0;
                end
            end
            
            for j=1:32
                if strcmp(dp_bi(j),'1')
                    out(j)=1;
                else
                    out(j)=0;
                end
            end
            
        end
        function w = randomizeWeights(obj)
            %randomize the controller constants to a scale (must be positive)
            if obj.binary==false
                for j=1:length(obj.gene)
                    obj.gene(j)=(rand*(obj.maxes(j)-obj.mins(j)))+obj.mins(j);
                end
            else
                obj.gene=[];
                for j=1:length(obj.maxes);
                    obj.gene=[obj.gene obj.real2bi((rand*(obj.maxes(j)-obj.mins(j)))+obj.mins(j))];
                end
            end
            w=obj.gene;
        end
        
        
        function w = setWeights(obj,in)
            %set constants
            if obj.binary==false
                obj.gene=in;
                for j=1:length(obj.gene)
                    if obj.gene(j)>obj.maxes(j)
                        obj.gene(j)=obj.maxes(j);
                    elseif obj.gene(j)<obj.mins(j)
                        obj.gene(j)=obj.mins(j);
                    end
                end
            else
               obj.gene=in;
               realgene=obj.real_conv(obj.gene);
               for j=1:length(realgene)
                    if realgene(j)>obj.maxes(j)
                        realgene(j)=obj.maxes(j);
                    elseif realgene(j)<obj.mins(j)
                        realgene(j)=obj.mins(j);
                    end
               end
               obj.gene=real2bi(realgene);
            end
            w=obj.gene;
        end
        
        function w = getWeights(obj)
            w=obj.gene;
        end
    end
end