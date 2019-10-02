function data = AddItem( data, train_set )
if numel(size(data.buffer)) == 2
    data.buffer(:, data.index) = train_set;
    if data.index == data.maxIndex
        data.index = 1;
    else
        data.index = data.index + 1;
    end
elseif numel(size(data.buffer)) == 3
    data.buffer(:,:, data.index) = train_set;
    if data.index == data.maxIndex
        data.index = 1;
    else
        data.index = data.index + 1;
    end
end
end

