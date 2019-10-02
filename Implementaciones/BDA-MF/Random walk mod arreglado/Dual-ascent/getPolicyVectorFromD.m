function policy = getPolicyVectorFromD( d, game )
%GETDNORMALIZED Obtains the current policy from vector d of joint
%probability distribution 

sumDoverA_aux = sum(reshape(d,[game.numActions,game.numStates]),1);
sumDoverA = sum(game.duplicar*diag(sumDoverA_aux),2);
policy = d./sumDoverA;

end

