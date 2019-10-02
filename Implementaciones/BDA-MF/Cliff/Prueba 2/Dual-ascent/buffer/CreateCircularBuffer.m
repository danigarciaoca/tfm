function data = CreateCircularBuffer(numRow, numCol)
data.index = 1;
data.maxIndex = numRow;
data.s_a_sNext = zeros(numRow, numCol);
end

