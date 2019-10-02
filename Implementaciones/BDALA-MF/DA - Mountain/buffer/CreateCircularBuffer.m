function data = CreateCircularBuffer(numRow, numCol)
data.index = 1;
data.maxIndex = numRow;
data.array = zeros(numRow, numCol);
end

