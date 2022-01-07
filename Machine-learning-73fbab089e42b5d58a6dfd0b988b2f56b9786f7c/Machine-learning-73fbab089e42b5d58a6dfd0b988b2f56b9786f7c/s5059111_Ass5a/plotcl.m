function plotcl(X, Xlbl, coord)
    colors = 'gbrkyrmcgbwk';
    pointstyles = 'o+*xsd^v><ph'; 
    
    if nargin<3
        coord=[1 2 3];
    end
    for h=1:max(Xlbl)
        cc = colors(mod(h-1,length(colors))+1);
        ps = pointstyles(mod(h-1,length(pointstyles))+1);
        select = Xlbl==h;
        if size(X,2)==1
            plot(X(select,1),X(select,1),[ps cc]);
        elseif size(X,2)==2
            plot(X(select,1),X(select,2),[ps cc]);
        else
            plot3(X(select,coord(1)),X(select,coord(2)),X(select,coord(3)),[ps cc]);
        end
        hold on
    end
    hold off
end


