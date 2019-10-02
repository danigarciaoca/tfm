function policy = getPolicyVectorFromD( d, game )
%GETDNORMALIZED Obtains the current policy from vector d of joint
%probability distribution 

sumDoverA_aux = sum(reshape(d,[game.num_actions,game.S]),1);
sumDoverA = sum(game.duplicate*diag(sumDoverA_aux),2);
policy = d./sumDoverA;

end

