import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from numpy.polynomial.polynomial import polydiv, polyval
import skrf
from NullSpaceMethod import data_handler, complex_to_real_matrix, sturm_count_zeros, myfunction_mm

np.random.seed(0)#replicate results



def main():
    filepath=r"C:\Users\ballo\OneDrive\Desktop\Code\Independent Study\Version 7 compare to vector fit\4310lc\4310LC-132_series.csv"
    data=load_sparameter_data(filepath)
    d1=data['S21']
    lengthD1=len(d1)
    print(f"Dataset Size: {lengthD1}")
    
    y_min=d1.iloc[:,1].min()
    y_max=d1.iloc[:,1].max()
    print("\nDataset loaded successfully.")
    print(f"S21 range: {y_min} - {y_max}")
    
    n_points=len(d1)
    
    MSE_THRESHOLD=1e-2
    MAX_ERROR_THRESHOLD=1e-3
    MAX_REFINEMENTS=6  #More iterations for random sampling exploration
    TARGET_POINTS=100     #Subsample refined data to this size, helpful for large datasets and speed, while mantaining shape
    
    #Fixed degree range for all iterations
    degree_range=range(70,90,1)
    
    best_per_iteration=[]
    found_zero_spike=None
    
    #Run on original data first
    print(f"\n\nTrying Original Data ({n_points} points)")
    
    results,zero_spike=evaluate_degree_range(d1,degree_range,"Original Data",0,MSE_THRESHOLD)
    best,found_zero_spike=process_results(results,zero_spike,found_zero_spike,
                                              best_per_iteration,MSE_THRESHOLD,MAX_ERROR_THRESHOLD)
    
    #Run midpoint refinements with subsampling
    subdivisions=1
    for i in range(1,MAX_REFINEMENTS+1):
        if found_zero_spike is not None:
            print(f"\n  ZERO-SPIKE FOUND - stopping refinements  ")
            break
        
        #Always refine from ORIGINAL data, not from previous subsample
        refined_df=MidpointRefine(d1.copy(),subdivisions=subdivisions)
        refined_n=len(refined_df)
        
        if refined_n>TARGET_POINTS:
            step=refined_n//TARGET_POINTS
            #First 2 iterations use offset-based, rest use random
            if i<=2:
                offset=(i*step)//(MAX_REFINEMENTS+1)
                current_df=subsample_data(refined_df,step,offset)
            else:
                current_df=subsample_data(refined_df,step,random_mode=True,seed=i*42)
        else:
            current_df=refined_df
        
        working_n=len(current_df)
        
        print(f"\n{'='*60}")
        print(f"TRYING Refinement {i} ({working_n} points after subsample)")
        print(f"{'='*60}")
        
        results,zero_spike=evaluate_degree_range(current_df,degree_range,f"Refinement_{i}",i,MSE_THRESHOLD)
        _,found_zero_spike=process_results(results,zero_spike,found_zero_spike,
                                              best_per_iteration,MSE_THRESHOLD,MAX_ERROR_THRESHOLD)
        subdivisions*=2
    
    #Display comparison
    print_comparison_table(best_per_iteration)
    
    #Final Selection - pass original df for full plot
    select_and_display_winner(best_per_iteration,found_zero_spike,d1)


def load_sparameter_data(filepath):
    #Skip the first two lines
    df=pd.read_csv(filepath,comment="!",skiprows=2,header=None,sep=",",       # the file is comma-separated, skips the 2 initial ! lines
        names=["freq_Hz","S11r","S11i","S12r","S12i","S21r","S21i","S22r","S22i",])
    df=df.dropna()

    #Convert columns to numeric
    df=df.astype(float)
    #Frequency in Hz
    freq=df["freq_Hz"].to_numpy()
    
    S21=df["S21r"].to_numpy()+1j*df["S21i"].to_numpy()
    S22=df["S22r"].to_numpy()+1j*df["S22i"].to_numpy()

    # Return as a tuple of two DataFrames (S21 and S22)
    df_S21=pd.DataFrame({"Frequency_Hz":freq,"S21_complex":S21})
    df_S22=pd.DataFrame({"Frequency_Hz":freq,"S22_complex":S22})
    return {"S21":df_S21,"S22":df_S22}

def subsample_data(df,step,offset=0,random_mode=False,seed=None):
    #Take every Nth point with optional offset or random sampling
    f=df.iloc[:,0].to_numpy(float)
    n=len(f)
    
    if random_mode and seed is not None:
        #Random subsampling - pick ~n/step points randomly
        np.random.seed(seed)
        n_samples=max(n//step,10)
        indices=sorted(np.random.choice(n,size=n_samples,replace=False).tolist())
        #Ensure first and last are included
        if 0 not in indices:
            indices.insert(0,0)
        if n-1 not in indices:
            indices.append(n-1)
        indices=sorted(set(indices))
        print(f"Random subsampled: {n} -> {len(indices)} points (seed {seed})")
    else:
        #Regular subsampling with offset
        start=min(offset,step-1) if step>1 else 0
        indices=list(range(start,n,step))
        
        #Ensure first and last points are included
        if 0 not in indices:
            indices.insert(0,0)
        if n-1 not in indices:
            indices.append(n-1)
        
        indices=sorted(set(indices))
        print(f"Subsampled: {n} -> {len(indices)} points (every {step}th, offset {start})")
    
    new_df=df.iloc[indices].copy().reset_index(drop=True)
    new_df.columns=[0,1]
    return new_df


def process_results(results,zero_spike_model,found_zero_spike,best_per_iteration,mse_thresh,max_err_thresh):
    valid=[r for r in results 
             if not np.isinf(r['sse']) 
             and r['mse']<mse_thresh 
             and r['max_error']<max_err_thresh]
    
    best=None
    if valid:
        best=min(valid,key=lambda x:(x['spike_score'],x['mse']))
        best_per_iteration.append(best)
    
    if zero_spike_model is not None and found_zero_spike is None:
        found_zero_spike=zero_spike_model
    
    return best,found_zero_spike


def print_comparison_table(best_per_iteration):
    print("COMPARISON OF BEST MODEL FROM EACH ITERATION")
    print(f"{'Source':<20} | {'Degree':<8} | {'Spikes':<8} | {'MSE':<14} | {'Max Error':<14}")
    print("-"*70)
    for model in best_per_iteration:
        print(f"{model['source']:<20} | {model['degree']:<8} | {model['spike_score']:<8.0f} | {model['mse']:<14.4e} | {model['max_error']:<14.4e}")
    print("-"*70)


def select_and_display_winner(best_per_iteration,found_zero_spike,original_df):
    if found_zero_spike is not None:
        print(f"\nZERO-SPIKE SOLUTION FOUND!")
        print_model_stats(found_zero_spike)
        plot_final_model(found_zero_spike,original_df)
        return
    
    print("\nNO ZERO-SPIKE SOLUTION FOUND")
    PERFECT_FIT_TOLERANCE=1e-8 
    perfect_fit_candidates=[m for m in best_per_iteration if m['max_error']<PERFECT_FIT_TOLERANCE]
    
    if perfect_fit_candidates:
        print(f"\nFound {len(perfect_fit_candidates)} model(s)")
        best_model=min(perfect_fit_candidates,key=lambda x:(x['spike_score'],x['mse']))
    else:
        print("\nNo numerically 'perfect' fits found. Selecting by minimum spikes.")
        best_model=min(best_per_iteration,key=lambda x:(x['spike_score'],x['mse']))
    
    print(f"\nWINNING MODEL SELECTED:")
    print_model_stats(best_model)
    plot_final_model(best_model,original_df)


def print_model_stats(model):
    print(f"Source: {model['source']}")
    print(f"Degree: {model['degree']}")
    print(f"Spike Score: {model['spike_score']}")
    print(f"MSE: {model['mse']:.6e}")
    print(f"Max Error: {model['max_error']:.6e}")


def count_spikes_and_errors(df,m):
    f=df.iloc[:,0].to_numpy(dtype=float)
    X=df.iloc[:,1].to_numpy(dtype=complex)
    N=len(f)

    f_min,f_max=np.min(f),np.max(f)
    f_norm=2*(f-f_min)/(f_max-f_min)-1

    A_complex=np.zeros((N,2*m+2),dtype=complex)
    for k in range(N):
        for i in range(m+1):
            A_complex[k,i]=-f_norm[k]**i
            A_complex[k,i+m+1]=X[k]*(f_norm[k]**i)

    A=complex_to_real_matrix(A_complex)
    x_all=null_space(A).T
    if x_all.size==0:
        return {"spike_score":np.inf,"max_error":np.inf,"sse":np.inf,"mse":np.inf}

    ysv=np.zeros((x_all.shape[0],N),dtype=complex)
    for j in range(x_all.shape[0]):
        for i in range(N):
            ysv[j,i]=myfunction_mm(x_all[j,:],f_norm[i],m)

    myerr=ysv-X
    sse_all=np.sum(np.abs(myerr)**2,axis=1)
    maxerr_all=np.max(np.abs(myerr),axis=1)
    mse=sse_all/N

    I=np.argmin(sse_all)
    best_x=x_all[I,:]

    L=len(best_x)//2
    x_complex=best_x[:L]+1j*best_x[L:]
    num=x_complex[:m+1]
    den=x_complex[m+1:]

    #Convert to real polynomial coefficients (denominator spikes)
    den_real=np.real(den)

    #Use Sturm's theorem to count number of distinct real poles in [-1,1]
    #Note THIS ONE MUST USE [-1,1] since the values can be negative in s-parameters
    spikes=sturm_count_zeros(den_real,-1,1)
    spikes=max(spikes,0)
    
    #Dense grid visual spike detection
    y_min,y_max=np.min(np.abs(X)),np.max(np.abs(X))
    y_range=y_max-y_min
    tolerance=2.0  #Allow 2x the data range as margin
    bound_low=y_min-tolerance*y_range
    bound_high=y_max+tolerance*y_range
    
    f_dense=np.linspace(f_norm.min(),f_norm.max(),500)
    pred_dense=np.array([myfunction_mm(best_x,s,m) for s in f_dense])
    pred_dense_abs=np.abs(pred_dense)
    
    #Count how many dense points exceed bounds
    visual_spikes=np.sum((pred_dense_abs<bound_low)|(pred_dense_abs>bound_high))
    
    #Add visual spikes as penalty (scaled down since there are 500 test points)
    total_spike_score=spikes+visual_spikes//10

    return {"spike_score":total_spike_score,"max_error":maxerr_all[I],"sse":sse_all[I],"mse":mse[I]}


#Used to regenerate coefficients for the final plot
def get_coefficients_and_stats(df,m):
    f=df.iloc[:,0].to_numpy(dtype=float)
    X=df.iloc[:,1].to_numpy(dtype=complex)
    N=len(f)

    f_min,f_max=np.min(f),np.max(f)
    f_norm=2*(f-f_min)/(f_max-f_min)-1

    A_complex=np.zeros((N,2*m+2),dtype=complex)
    for k in range(N):
        for i in range(m+1):
            A_complex[k,i]=-f_norm[k]**i
            A_complex[k,i+m+1]=X[k]*(f_norm[k]**i)

    A=complex_to_real_matrix(A_complex)
    x_all=null_space(A).T
    if x_all.size==0:
        return None,{"spike_score":np.inf,"max_error":np.inf,"sse":np.inf,"mse":np.inf}

    ysv=np.zeros((x_all.shape[0],N),dtype=complex)
    for j in range(x_all.shape[0]):
        for i in range(N):
            ysv[j,i]=myfunction_mm(x_all[j,:],f_norm[i],m)

    myerr=ysv-X
    sse_all=np.sum(np.abs(myerr)**2,axis=1)
    maxerr_all=np.max(np.abs(myerr),axis=1)
    mse=sse_all/N

    I=np.argmin(sse_all)
    best_x=x_all[I,:]
    
    L=len(best_x)//2
    x_complex=best_x[:L]+1j*best_x[L:]
    
    den=x_complex[m+1:]
    den_real=np.real(den)
    spikes=sturm_count_zeros(den_real,-1,1)

    stats={"spike_score":spikes,"max_error":maxerr_all[I],"sse":sse_all[I],"mse":mse[I]}
    return best_x,stats


def evaluate_degree_range(df,degree_range,source_name,attempt_num,mse_threshold=1e-2):
    results=[]
    zero_spike_model=None
    #Print Header for this run
    print(f"\nRUN: {source_name} (Attempt {attempt_num})")
    print(f"{'Degree':<8} | {'Spikes':<8} | {'MSE':<12} | {'Max Error':<12}")
    print(f"{'-'*46}")
    
    for m in degree_range:
        stats=count_spikes_and_errors(df,m)
        
        #Print every tested degree
        print(f"{m:<8} | {stats['spike_score']:<8} | {stats['mse']:.4e}   | {stats['max_error']:.4e}")
        
        entry={"spike_score":stats["spike_score"],"sse":stats["sse"],"mse":stats["mse"],
            "max_error":stats["max_error"],"degree":m,"source":source_name,"attempts":attempt_num,"dataset":df.copy()}
        results.append(entry)
        
        #Early stop on 0-spike with reasonable MSE (relaxed threshold)
        if stats["spike_score"]==0 and zero_spike_model is None and stats["mse"]<1.0:
            zero_spike_model=entry
            print(f"\n 0-SPIKE FOUND at degree {m}!")
            break
        
    print(f"{'-'*46}")
    return results,zero_spike_model


def MidpointRefine(df,subdivisions=1):
    if subdivisions<1: 
        return df.copy() #if for some reason subdivisions less than one 
    
    f=df.iloc[:,0].to_numpy(float)
    X=df.iloc[:,1].to_numpy(complex)
    new_f=[]
    new_X=[]
    
    for i in range(len(f)-1): #add specified number of midpoints
        a_f,b_f=f[i],f[i+1]
        a_X,b_X=X[i],X[i+1]
        new_f.append(a_f)
        new_X.append(a_X)
        for s in range(1,subdivisions+1):
            t=s/(subdivisions+1)
            new_f.append(a_f*(1-t)+b_f*t)
            new_X.append(a_X*(1-t)+b_X*t)
    new_f.append(f[-1])
    new_X.append(X[-1])
    
    out=pd.DataFrame({0:np.array(new_f),1:np.array(new_X)})
    out=out.sort_values(0).reset_index(drop=True)
    print(f"Dataset expanded New size: {len(out)} points.")
    return out


def plot_final_model(model,original_df):
    #Plot winning model against data
    fx,_=get_coefficients_and_stats(model['dataset'],model['degree'])
    if fx is None:
        return
    
    #Original data
    f_orig=original_df.iloc[:,0].to_numpy(float)
    X_orig=original_df.iloc[:,1].to_numpy(complex)
    
    #Subset data (model was trained on this)
    df_subset=model['dataset']
    f_subset=df_subset.iloc[:,0].to_numpy(float)
    f_min,f_max=np.min(f_subset),np.max(f_subset)
    f_subset_norm=2*(f_subset-f_min)/(f_max-f_min)-1
    
    #Curve for plotting (evaluated on subset points - the stable fit)
    pred_subset=np.array([myfunction_mm(fx,s,model['degree']) for s in f_subset_norm])
    
    #Interpolate predictions at original x-values (safer than direct polynomial eval)
    pred_orig_real=np.interp(f_orig,f_subset,np.real(pred_subset))
    pred_orig_imag=np.interp(f_orig,f_subset,np.imag(pred_subset))
    pred_orig=pred_orig_real+1j*pred_orig_imag
    
    #Calculate errors vs original data
    errors=np.abs(pred_orig-X_orig)
    mse_orig=np.mean(errors**2)
    sse_ns=np.sum(errors**2)
    max_err=np.max(errors)
    min_err=np.min(errors)
    
    #VectorFitting comparison
    freq_hz=f_orig
    s_matrix=X_orig.reshape(-1,1,1)
    nw=skrf.Network(frequency=skrf.Frequency.from_f(freq_hz,unit='Hz'),s=s_matrix,name='S21')
    
    vf=skrf.VectorFitting(nw)
    try:
        vf.vector_fit(n_poles_real=3,n_poles_cmplx=10,init_pole_spacing='lin')
        H_vf=vf.get_model_response(0,0,nw.frequency.f)
        err_vf=np.abs(H_vf-X_orig)
        sse_vf=np.sum(err_vf**2)
        mse_vf=np.mean(err_vf**2)
        vf_success=True
    except Exception as e:
        print(f"VectorFitting failed: {e}")
        vf_success=False
        H_vf=None
        sse_vf=np.inf
    
    #Print consolidated results
    print("\n\nERROR ANALYSIS VS ORIGINAL DATA")
    print(f"{'='*60}")
    print(f"Points: {len(f_orig)} | NS-SSE: {sse_ns:.4e} | NS-MSE: {mse_orig:.4e}")
    print(f"Max Err: {max_err:.4e} | Min Err: {min_err:.4e}")
    if vf_success:
        ratio=sse_ns/sse_vf if sse_vf>0 else np.inf
        winner="Null-Space" if ratio<1 else "VectorFit"
        print(f"\nVF-SSE: {sse_vf:.4e} | VF-MSE: {mse_vf:.4e}")
        print(f"Ratio (NS/VF): {ratio:.3f} → {winner} wins!")
    print(f"{'='*60}\n")
    
    #Convert frequency to GHz for plotting
    freq_GHz=f_orig/1e9
    
    #Create figure with 2 subplots
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(12,10))
    
    #Plot 1: Magnitude comparison
    ax1.plot(freq_GHz,20*np.log10(np.abs(X_orig)),'ro',markersize=3,label=f"Original Data ({len(f_orig)} pts)")
    ax1.plot(freq_GHz,20*np.log10(np.abs(pred_orig)),'b-',linewidth=2,label=f"Null-Space (deg {model['degree']}, {int(model['spike_score'])} spikes)")
    if vf_success:
        ax1.plot(freq_GHz,20*np.log10(np.abs(H_vf)),'g--',linewidth=2,label='VectorFitting')
    ax1.set_title(f"FINAL MODEL: {model['source']} - Magnitude Comparison")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.grid(True)
    ax1.legend()
    
    #Plot 2: Error comparison
    ax2.plot(freq_GHz,errors,'b-',linewidth=1.5,label=f"Null-Space Error (SSE={sse_ns:.4e})")
    if vf_success:
        ax2.plot(freq_GHz,err_vf,'g-',linewidth=1.5,label=f"VectorFit Error (SSE={sse_vf:.4e})")
    ax2.axhline(y=mse_orig**0.5,color='b',linestyle='--',alpha=0.5,label=f"NS RMSE={mse_orig**0.5:.4e}")
    ax2.set_title("Error Comparison: Null-Space vs VectorFitting")
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()