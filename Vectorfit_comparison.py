import numpy as np
import pandas as pd
import skrf
import matplotlib.pyplot as plt
from scipy.linalg import null_space
from numpy.polynomial import polynomial as P
from utils import sturm_count_zeros
import sys

def main():
    filepath=r"C:\Users\ballo\OneDrive\Desktop\Code\Independent Study\Version 12 Improving the vector fit\4310LC-132_series.csv"
    pole_counts_null=list(range(30, 100, 5))
    pole_counts=list(range(2, 30, 2))
    
    freq, s21=load_data(filepath)
    nw=create_network(freq, s21)
    
    vf_results=test_vector_fitting(nw, pole_counts, freq, s21)
    ns_results,_=bootstrap_nullspace(freq, s21, pole_counts_null)
    
    print_comparison(vf_results, ns_results)
    plot_comparison(vf_results, ns_results, freq, s21)

def load_data(filepath):
    df=pd.read_csv(filepath, comment='!', header=None, dtype=str)
    df=df[~df.iloc[:, 0].str.startswith('#')].astype(float)
    
    freq=df.iloc[:, 0].to_numpy()
    mag=10**(df.iloc[:, 3].to_numpy()/20.0)
    phase=np.deg2rad(df.iloc[:, 4].to_numpy())
    return freq, mag*np.exp(1j*phase)

def create_network(freq, s_param):
    s_matrix=np.zeros((len(freq), 2, 2), dtype=complex)
    s_matrix[:, 1, 0]=s_param
    return skrf.Network(frequency=skrf.Frequency.from_f(freq, unit="Hz"), s=s_matrix)

def count_spikes_sturm(poly_coeffs, a=-1, b=1):
    poly_real=np.real(poly_coeffs)
    try:
        spike_count=sturm_count_zeros(poly_real, a, b)
        return max(spike_count, 0)
    except:
        return None


def test_vector_fitting(nw, pole_counts, freq, s_original):
    results=[]
    print("\nVECTOR FITTING RESULTS")
    print("="*70)
    
    for n_poles in pole_counts:
        try:
            vf=skrf.VectorFitting(nw)
            vf.vector_fit(n_poles_real=n_poles,n_poles_cmplx=0)
            
            fitted=vf.get_model_response(1, 0, freq)
            error=np.abs(s_original-fitted)
            spike_count=None
            
            results.append({'n_poles': n_poles,'rms_error': vf.get_rms_error(),
                'max_error': np.max(error),'mean_error': np.mean(error),'rss_error': np.sum(error**2),'fitted_response': fitted,
                'vf': vf,'poles': vf.poles,'residues': vf.residues,'spike_count': spike_count})
            
            spike_str=f" | Spikes: {spike_count}" if spike_count is not None else ""
            print(f"{n_poles:<6} | RMS: {vf.get_rms_error():.6f} | Max: {np.max(error):.6f}{spike_str}")
            
        except Exception as e:
            print(f"{n_poles:<6} | Failed: {str(e)}")
    
    return results

#NULLSPACE METHOD do not touch when at all possible
def myfunction_mm(x, s, m):
    L=len(x)//2
    x_complex=x[:L]+1j*x[L:]
    rexp=np.array([s**j for j in range(m+1)], dtype=complex)
    top=np.dot(x_complex[:m+1], rexp)
    bottom=np.dot(x_complex[m+1:], rexp)
    return top/(bottom if np.abs(bottom)>1e-12 else 1e-12)

def fit_nullspace_model(freq, s_param, m):
    N=len(freq)
    f_min, f_max=np.min(freq), np.max(freq)
    f_norm=2*(freq-f_min)/(f_max-f_min)-1
    
    A_complex=np.zeros((N, 2*m+2), dtype=complex)
    for k in range(N):
        for i in range(m+1):
            A_complex[k, i]=-f_norm[k]**i
            A_complex[k, i+m+1]=s_param[k]*(f_norm[k]**i)
    
    A_real=np.block([[np.real(A_complex), -np.imag(A_complex)],
                     [np.imag(A_complex), np.real(A_complex)]])
    x=null_space(A_real).T
    
    if x.size==0:
        return None, None, None
    
    ysv=np.array([[myfunction_mm(x[j, :], f_norm[i], m) for i in range(N)] for j in range(x.shape[0])])
    myerr=ysv-s_param
    I=np.argmin(np.max(np.abs(myerr), axis=1))
    
    L=len(x[I, :])//2
    coeffs_complex=x[I, :L]+1j*x[I, L:]
    numerator_coeffs=coeffs_complex[:m+1]
    denominator_coeffs=coeffs_complex[m+1:]
    
    return ysv[I, :], myerr[I, :], (numerator_coeffs, denominator_coeffs)

#RESULTS
def print_comparison(vf_results, ns_results):
    print("\nCOMPARISON SUMMARY")
    print("="*70)
    
    if vf_results:
        best=min(vf_results, key=lambda x: x['rms_error'])
        print(f"\nBest Vector Fitting: {best['n_poles']} poles")
        print(f"  RMS: {best['rms_error']:.6f} | Max: {best['max_error']:.6f} | RSS: {best['rss_error']:.6e} | Spikes: {best['spike_count']}")
    
    if ns_results: #picks out best based on spike count first, then rss error
        best = min(ns_results, key=lambda x: (x['spike_count'] if x['spike_count'] is not None else np.inf,x['rss_error']))
        spike_info=f" | Spikes: {best['spike_count']}" if best.get('spike_count') is not None else ""
        print(f"\nBest Nullspace: Degree {best['degree']}")
        print(f"  Max: {best['max_error']:.6f} | Mean: {best['mean_error']:.6f} | RSS: {best['rss_error']:.6e}{spike_info}")

def plot_comparison(vf_results, ns_results, freq, s_original):
    fig, axes=plt.subplots(2, 3, figsize=(18, 10))
    
    for i, (key, title) in enumerate([('max_error', 'Max Error'), ('mean_error', 'Mean Error'), ('rss_error', 'RSS Error')]):
        if vf_results:
            axes[0, i].plot([r['n_poles'] for r in vf_results], [r[key] for r in vf_results], 'bo-', label='VF')
        if ns_results:
            axes[0, i].plot([r['degree'] for r in ns_results], [r[key] for r in ns_results], 'rs-', label='NS')
        axes[0, i].set_xlabel('Poles/Degree')
        axes[0, i].set_ylabel(title)
        axes[0, i].set_title(title)
        axes[0, i].legend()
        axes[0, i].grid(True)
        if i==2:
            axes[0, i].set_yscale('log')
    
    if vf_results:
        best = min(vf_results, key=lambda x: x['rms_error'])
        axes[1, 0].plot(freq/1e9, 20*np.log10(np.abs(s_original)), 'b-', label='Original')
        axes[1, 0].plot(freq/1e9, 20*np.log10(np.abs(best['fitted_response'])), 'r--', label=f"{best['n_poles']} poles")
        axes[1, 0].set_title('Vector Fitting - Best')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    if ns_results:
        best = min(ns_results, key=lambda x: (x['spike_count'] if x['spike_count'] is not None else np.inf,x['rss_error']))
        fitted_freq=best['fitted_freq']
        axes[1, 1].plot(fitted_freq/1e9, 20*np.log10(np.abs(best['fitted_response'])), 'r--', label=f"Deg {best['degree']}")
        axes[1, 1].plot(freq/1e9, 20*np.log10(np.abs(s_original)), 'b-', label='Original', alpha=0.7)
        axes[1, 1].set_title('Nullspace - Best')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    axes[1, 2].plot(freq/1e9, 20*np.log10(np.abs(s_original)), 'k-', label='Original', linewidth=2.5)
    if vf_results:
        best=min(vf_results, key=lambda x: x['rms_error'])
        axes[1, 2].plot(freq/1e9, 20*np.log10(np.abs(best['fitted_response'])), 'b--', label=f"VF {best['n_poles']}")
    if ns_results:
        best = min(ns_results, key=lambda x: (x['spike_count'] if x['spike_count'] is not None else np.inf,x['rss_error']))
        fitted_freq=best['fitted_freq']
        axes[1, 2].plot(fitted_freq/1e9, 20*np.log10(np.abs(best['fitted_response'])), 'r:', label=f"NS {best['degree']}")
    axes[1, 2].set_title('Best Methods Comparison')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    for ax in axes[1, :]:
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Magnitude (dB)')
    
    plt.tight_layout()
    plt.show()

def subsample_data(df, step, offset=0, random_mode=False, seed=None):
    f=df.iloc[:,0].to_numpy(float)
    n=len(f)
    
    if random_mode and seed is not None:
        np.random.seed(seed)
        n_samples=max(n//step,10)
        indices=sorted(np.random.choice(n,size=n_samples,replace=False).tolist())
        if 0 not in indices:
            indices.insert(0,0)
        if n-1 not in indices:
            indices.append(n-1)
        indices=sorted(set(indices))
    else:
        start=min(offset,step-1) if step>1 else 0
        indices=list(range(start,n,step))
        if 0 not in indices:
            indices.insert(0,0)
        if n-1 not in indices:
            indices.append(n-1)
        indices=sorted(set(indices))
    
    new_df=df.iloc[indices].copy().reset_index(drop=True)
    new_df.columns=[0,1]
    return new_df

def MidpointRefine(df, subdivisions=1):
    if subdivisions < 1:
        return df.copy()
    
    f=df.iloc[:,0].to_numpy(float)
    X=df.iloc[:,1].to_numpy(complex)
    new_f=[]
    new_X=[]
    
    for i in range(len(f)-1):
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
    return out
def bootstrap_nullspace(freq, s_param, degrees, max_refinements=6, target_points=160):
    base_df=pd.DataFrame({0:freq,1:s_param})
    best_models=[]
    found_zero_spike=None
    subdivisions=1
    
    print("\n" + "="*70)
    print("NULLSPACE METHOD RESULTS")
    print("="*70)
    
    for i in range(max_refinements+1):
        if found_zero_spike is not None:
            break
        
        if i==0:
            working_df=base_df.copy()
            source="Original"
            print(f"\n[Round 0: Original Data - {len(working_df)} points]")
        else:
            refined=MidpointRefine(base_df.copy(),subdivisions=subdivisions)
            if len(refined)>target_points:
                step=len(refined)//target_points
                if i<=2:
                    offset=(i*step)//(max_refinements+1)
                    working_df=subsample_data(refined,step,offset)
                else:
                    working_df=subsample_data(refined,step,random_mode=True,seed=i*42)
            else:
                working_df=refined
            source=f"Refinement_{i}"
            print(f"\n[Round {i}]")
            subdivisions*=2
        
        f=working_df.iloc[:,0].to_numpy(float)
        X=working_df.iloc[:,1].to_numpy(complex)
        
        for m in degrees:
            print(f"Degree {m}...", end='')
            sys.stdout.flush()
            
            try:
                fitted, error, coeffs=fit_nullspace_model(f,X,m)
                if fitted is None:
                    print(" Failed: Empty null space")
                    continue
                
                err=np.abs(X-fitted)
                spikes=count_spikes_sturm(coeffs[1],-1,1)
                rss=np.sum(err**2)
                max_err=np.max(err)
                mean_err=np.mean(err)
                
                entry={"degree":m,"rss_error":rss,"max_error":max_err, "mean_error":mean_err,
                    "spike_count":spikes,"source":source,"fitted_response":fitted,"fitted_freq":f}
                best_models.append(entry)
                
                spike_str=f"Spikes: {spikes}" if spikes is not None else "Spikes: N/A"
                print(f" RSS: {rss:.6e} | Max: {max_err:.6f} | {spike_str}")
                
                if spikes==0 and found_zero_spike is None:
                    found_zero_spike=entry
                    print(f"\n0-SPIKE FOUND at degree {m} in Round {i}!")
                    break
            except Exception as e:
                print(f" Failed: {str(e)}")
    
    return best_models, found_zero_spike

if __name__=="__main__":
    main()