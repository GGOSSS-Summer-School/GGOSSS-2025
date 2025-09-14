%==========================================================================
%nemo_coastalvar_box select the variables in the selected box depends on the selected number of points
%of gridded data following the discrete scheme nemo model.
%The downward vertical integration of the water column(see nemo v4.2 page 25).
%function varargout = nemo_seaward_zonal_integration(varargin)
function [varout_indx,varout] = nemo_coastalvar_box(varargin)
%input variables
varin = varargin{1};% 2d or 3d or 4d (jpi,jpj,jpkm1,kt): selected variable
npoint = varargin{2};% number of selected points relative to the coast 
ninput = 2;% default: number of variables
%==========================================================================
if nargin >= ninput
    if nargin == ninput+1
        varin_coast = varargin{3};
    elseif nargin == ninput
        varin_coast = varin;
    end
    if size(varin,4) > 1%(jpi,jpj,jpkm1,kt)
        [jpi,jpj,jpkm1,kt] = size(varin);
        varout = nan*ones(jpi,jpj,jpkm1,kt);
        varout_indx = nan*ones(npoint,jpkm1,jpj);
        varin_mean = mean(varin_coast,4);
        %%select points in selected box
        for ik = 1:jpkm1
            varin_mean_ik = squeeze(varin_mean(:,:,ik));
            for ij = 1:jpj
                indx_coast_ik = max(find(isfinite(varin_mean_ik(:,ij))));
                if ~isempty(indx_coast_ik)
                    if indx_coast_ik-npoint+1 > 0
                        varout(indx_coast_ik-npoint+1:indx_coast_ik,ij,ik,:) = varin(indx_coast_ik-npoint+1:indx_coast_ik,ij,ik,:);
                        varout_indx(indx_coast_ik-npoint+1:indx_coast_ik,ij,ik) = (indx_coast_ik-npoint+1:indx_coast_ik)';
                    end
                end
            end%end for ij
        end%for ik
    elseif size(varin,3) > 1%(jpi,jpj,kt)
        [jpi,jpj,kt] = size(varin);
        varout = nan*ones(jpi,jpj,kt);
        varout_indx = nan*ones(npoint,jpj);
        varin_mean = mean(varin_coast,3);
        %%select points in selected box
        for ij = 1:jpj
            indx_coast = max(find(isfinite(varin_mean(:,ij))));
            if ~isempty(indx_coast)
                if indx_coast-npoint+1 > 0
                    varout(indx_coast-npoint+1:indx_coast,ij,:) = varin(indx_coast-npoint+1:indx_coast,ij,:);
                    varout_indx(indx_coast-npoint+1:indx_coast,ij) = (indx_coast-npoint+1:indx_coast)';
                end
            end
        end%end for ij
    elseif size(varin,2) > 1%(jpi,jpj)
       [jpi,jpj] = size(varin);
        varout = nan*ones(jpi,jpj);
        varout_indx = nan*ones(npoint,jpj);
        %%select points in selected box
        for ij = 1:jpj
            indx_coast = max(find(isfinite(varin_coast(:,ij))));
            if ~isempty(indx_coast)
                if indx_coast-npoint+1 > 0
                    varout(indx_coast-npoint+1:indx_coast,ij) = varin(indx_coast-npoint+1:indx_coast,ij);
                    varout_indx(indx_coast-npoint+1:indx_coast,ij) = (indx_coast-npoint+1:indx_coast)';
                end
            end
        end%end for ij
    else
        disp('error: check input variable or format (4d, 3d and 1d)')
    end
    %varargout = {varout};
    %varargout = varout;
else
    disp(['error: check number input (must be ',num2str(ninput),' or more)'])
end
%==========================================================================
end
