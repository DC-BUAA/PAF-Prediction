function [Result_Vertical,Result_Vertical_Completely] = Struct_to_Vertical(s, Title, path)

Result_Vertical = {};

fields = fieldnames(s);


for i = 1:length(fields)
    field = fields{i};

    if isempty(path)
        currentPath = [Title, '.', field]; 
    else
        currentPath = [path, '.', field]; 
    end
    

    value = s.(field);
    

    if isstruct(value)
        Result_Vertical = [Result_Vertical; Struct_to_Vertical(value, Title, currentPath)];
    else
  
        Result_Vertical = [Result_Vertical; {currentPath, value}];
    end
end



expandedData = {};
nameCount = containers.Map('KeyType', 'char', 'ValueType', 'int32'); 

for i = 1:size(Result_Vertical, 1)
    name = Result_Vertical{i, 1}; 
    value = Result_Vertical{i, 2}; 

    if iscell(value) || ismatrix(value) 
        
        if ismatrix(value)
            value = value(:); 
        end
        
        if isKey(nameCount, name)
            nameCount(name) = nameCount(name) + 1; 
        else
            nameCount(name) = 1; 
        end
        for j = 1:length(value)
            
            currentCount = nameCount(name);

            newIndex = size(expandedData, 1) + 1; 
       
            if length(value) == 1
                expandedData{newIndex, 1} = name; 
            else
                expandedData{newIndex, 1} = sprintf('%s_%d', name, currentCount); 
            end
            expandedData{newIndex, 2} = value(j); 
            
            if length(value) > 1
                nameCount(name) = nameCount(name) + 1; 
            end
        end

    end
end
Result_Vertical_Completely = expandedData;

end