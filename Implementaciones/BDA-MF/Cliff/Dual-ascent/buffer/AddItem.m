function data = AddItem( data, train_set )
data.s_a_sNext(data.index, :) = train_set;
if data.index == data.maxIndex
    data.index = 1;
else
    data.index = data.index + 1;
end
end

