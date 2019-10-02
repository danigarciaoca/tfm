function data = AddItem( data, train_set )
data.array(data.index, :) = train_set;
if data.index == data.maxIndex
    data.index = 1;
else
    data.index = data.index + 1;
end
end

