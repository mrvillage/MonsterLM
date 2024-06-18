use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    process::Command,
    sync::Mutex,
};

use extendr_api::prelude::*;
use lmutils::r::QuantNorm;
use log::info;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
    },
    slice::{ParallelSlice, ParallelSliceMut},
};

const CHROMOSOMES: u8 = 22;

fn remove_rows(ncols: usize, data: &[f64], indices: &HashSet<usize>) -> Vec<f64> {
    data.par_chunks_exact(ncols)
        .enumerate()
        .filter_map(|(i, x)| if indices.contains(&i) { None } else { Some(x) })
        .flat_map(|x| x.par_iter().copied())
        .collect::<Vec<_>>()
}

/// @export
#[extendr]
fn monsterlm(dir: &str, env_type: &str) -> Robj {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().filter_or("MONSTERLM_LOG", "info"),
    )
    .try_init();

    if env_type != "continuous" && env_type != "dichotomous" {
        panic!("env_type must be either 'continuous' or 'dichotomous'");
    }
    let continuous = env_type == "continuous";

    let dir = PathBuf::from(dir);

    let outcomes = lmutils::File::from_path(dir.join("outcomes.rkyv.gz"))
        .unwrap()
        .read_matrix::<f64, _, _>(true)
        .unwrap();
    let exposures = lmutils::File::from_path(dir.join("exposures.rkyv.gz"))
        .unwrap()
        .read_matrix::<f64, _, _>(true)
        .unwrap();

    let age_mat = lmutils::File::from_path(dir.join("age.rkyv.gz"))
        .unwrap()
        .read_matrix::<f64, _, _>(true)
        .unwrap();
    let sex_mat = lmutils::File::from_path(dir.join("sex.rkyv.gz"))
        .unwrap()
        .read_matrix::<f64, _, _>(true)
        .unwrap();
    let pcs_mat = lmutils::File::from_path(dir.join("pcs.rkyv.gz"))
        .unwrap()
        .read_matrix::<f64, _, _>(true)
        .unwrap();

    let combos = (1..=outcomes.cols())
        .into_par_iter()
        .flat_map(|outcome| {
            (1..=exposures.cols())
                .into_par_iter()
                .map(|exposure| {
                    let pheno = &outcomes.data()
                        [outcomes.rows() * (outcome - 1)..(outcomes.rows() * outcome)];
                    let trait_name = &outcomes.colnames().unwrap()[outcome + 1];
                    let expo = &exposures.data()
                        [exposures.rows() * (exposure - 1)..(exposures.rows() * exposure)];
                    let exposure_name = &exposures.colnames().unwrap()[exposure + 1];
                    let nan_indices = pheno
                        .iter()
                        .enumerate()
                        .filter_map(|(i, x)| if x.is_nan() { Some(i) } else { None })
                        .chain(expo.iter().enumerate().filter_map(|(i, x)| {
                            if x.is_nan() {
                                Some(i)
                            } else {
                                None
                            }
                        }))
                        .collect::<HashSet<_>>();

                    let pheno = remove_rows(outcomes.cols(), pheno, &nan_indices);
                    let mut expo = remove_rows(exposures.cols(), expo, &nan_indices);
                    let age = remove_rows(age_mat.cols(), age_mat.data(), &nan_indices);
                    let age =
                        faer::mat::from_column_major_slice::<f64>(&age, age.len(), age_mat.cols());
                    let sex = remove_rows(sex_mat.cols(), sex_mat.data(), &nan_indices);
                    let sex =
                        faer::mat::from_column_major_slice::<f64>(&sex, sex.len(), sex_mat.cols());
                    let pcs = remove_rows(pcs_mat.cols(), pcs_mat.data(), &nan_indices);
                    let pcs =
                        faer::mat::from_column_major_slice::<f64>(&pcs, pcs.len(), pcs_mat.cols());

                    let mut columns = vec![
                        age.col(3).try_as_slice().unwrap(),
                        sex.col(3).try_as_slice().unwrap(),
                    ];
                    columns.extend((3..=22).map(|x| pcs.col(x).try_as_slice().unwrap()));
                    let columns = columns
                        .into_iter()
                        .flat_map(|x| x.iter().copied())
                        .collect::<Vec<_>>();
                    let stratification_matrix =
                        faer::mat::from_column_major_slice::<f64>(&columns, age.nrows(), 22);
                    let e_final = if continuous {
                        lmutils::r::rm_stratification(stratification_matrix, &expo)
                            .collect::<Vec<_>>()
                    } else {
                        lmutils::r::standardization(&mut expo);
                        expo
                    };

                    let p_final = lmutils::r::rm_stratification(stratification_matrix, &pheno)
                        .collect::<Vec<_>>();
                    let xs = faer::mat::from_column_major_slice::<f64>(&e_final, e_final.len(), 1);
                    let lmutils::r::Lm {
                        residuals: mut p_resid,
                        adj_r2: e_on_p,
                        ..
                    } = lmutils::r::lm(xs, &p_final);

                    if continuous {
                        p_resid = lmutils::r::rm_heteroscedasticity(p_resid, &e_final);
                    } else {
                        let mut unique = e_final.to_vec();
                        unique.sort_by(|a, b| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
                        });
                        unique.dedup();
                        for u in unique {
                            let indices = e_final
                                .iter()
                                .enumerate()
                                .filter_map(|(i, x)| if *x == u { Some(i) } else { None })
                                .collect::<Vec<_>>();
                            let q = (indices.iter().map(|&i| p_resid[i]))
                                .collect::<Vec<_>>()
                                .quant_norm()
                                .zip(indices.par_iter())
                                .collect::<Vec<_>>();
                            for (x, i) in q {
                                p_resid[*i] = x;
                            }
                        }
                    }

                    (
                        p_resid,
                        e_final,
                        trait_name,
                        exposure_name,
                        nan_indices,
                        e_on_p,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let single_set_results = (1..=CHROMOSOMES)
        .into_par_iter()
        .flat_map(|chr| {
            let block = lmutils::File::from_path(dir.join(format!("chr_{}.rkyv.gz", chr)))
                .unwrap()
                .read_matrix::<f64, _, _>(true)
                .unwrap();
            let ncols = block.cols();
            combos.par_iter().map(
                move |(p_resid, e_final, trait_name, exposure_name, nan_indices, e_on_p)| {
                    let p_resid =
                        faer::mat::from_column_major_slice::<f64>(p_resid, p_resid.len(), 1);

                    let mut block_data = remove_rows(ncols, block.data(), nan_indices);
                    let ndata = block_data.len();
                    let nrows = ndata / ncols;
                    // by moving this up here, we don't need to clone block_data
                    let mat = faer::mat::from_column_major_slice::<f64>(
                        &block_data,
                        block_data.len() / ncols,
                        ncols,
                    );
                    let g_r2 = lmutils::get_r2s(mat, p_resid)[0].adj_r2();

                    if continuous {
                        block_data.par_chunks_mut(ncols).for_each(|chunk| {
                            chunk
                                .iter_mut()
                                .zip(e_final.iter())
                                .for_each(|(g, e)| *g *= e);
                            let c = chunk.to_vec();
                            c.quant_norm().zip(chunk).for_each(|(q, g)| *g = q);
                        });
                    } else {
                        block_data.par_chunks_mut(ncols).for_each(|chunk| {
                            chunk
                                .iter_mut()
                                .zip(e_final.iter())
                                .for_each(|(g, e)| *g *= e);
                            lmutils::r::standardization(chunk);
                        });
                    }
                    let mat = faer::mat::from_column_major_slice::<f64>(&block_data, nrows, ncols);
                    let gxe_r2 = lmutils::get_r2s(mat, p_resid)[0].adj_r2();

                    struct SingleSetResult {
                        pub nb_indi: usize,
                        pub nb_snps: usize,
                        pub gxe_r2: f64,
                        pub g_r2: f64,
                        pub e_on_p: f64,
                        pub trait_name: String,
                        pub exposure_name: String,
                    }
                    SingleSetResult {
                        nb_indi: ndata,
                        nb_snps: ncols,
                        gxe_r2,
                        g_r2,
                        e_on_p: *e_on_p,
                        trait_name: trait_name.to_string(),
                        exposure_name: exposure_name.to_string(),
                    }
                },
            )
        })
        .collect::<Vec<_>>();

    let mut results_by_trait_and_exposure = HashMap::new();
    for i in single_set_results {
        let key = (i.trait_name.clone(), i.exposure_name.clone());
        results_by_trait_and_exposure
            .entry(key)
            .or_insert_with(Vec::new)
            .push(i);
    }

    let results = results_by_trait_and_exposure
        .into_iter()
        .flat_map(|((trait_name, exposure_name), r2s)| {
            #[allow(dead_code)]
            #[derive(Debug)]
            struct Results {
                pub model: &'static str,
                pub outcome: String,
                pub exposure: String,
                pub pred_n: usize,
                pub n: usize,
                pub est_adj: f64,
                pub lci: f64,
                pub uci: f64,
                pub variance_total: f64,
                pub standard_deviation: f64,
            }

            let e_on_p = r2s[0].e_on_p;

            let tot = r2s
                .iter()
                .map(|x| {
                    #[allow(unused)]
                    let adj_r2 = x.g_r2;
                    let p = x.nb_snps as f64;
                    let n = x.nb_indi as f64;
                    (1.0 - e_on_p)
                        * ((n - 1.0) / (n - p - 1.0)).powi(2)
                        * R!("Variance.R2({{adj_r2}}, {{n}}, {{p}})")
                            .unwrap()
                            .as_real()
                            .unwrap()
                })
                .sum::<f64>();
            let est_adj = (1.0 - e_on_p) * r2s.iter().map(|x| x.g_r2).sum::<f64>();
            let g_results = Results {
                model: "G",
                outcome: trait_name.clone(),
                exposure: exposure_name.clone(),
                pred_n: r2s.iter().map(|x| x.nb_snps).sum::<usize>(),
                n: r2s[0].nb_indi,
                est_adj,
                lci: est_adj - 1.96 * tot.sqrt(),
                uci: est_adj + 1.96 * tot.sqrt(),
                variance_total: tot,
                standard_deviation: tot.sqrt(),
            };

            let tot = r2s
                .iter()
                .map(|x| {
                    #[allow(unused)]
                    let adj_r2 = x.gxe_r2;
                    let p = x.nb_snps as f64;
                    let n = x.nb_indi as f64;
                    (1.0 - e_on_p)
                        * ((n - 1.0) / (n - p - 1.0)).powi(2)
                        * R!("Variance.R2({{adj_r2}}, {{n}}, {{p}})")
                            .unwrap()
                            .as_real()
                            .unwrap()
                })
                .sum::<f64>();
            let est_adj = (1.0 - e_on_p) * r2s.iter().map(|x| x.gxe_r2).sum::<f64>();
            let gxe_results = Results {
                model: "GxE",
                outcome: trait_name,
                exposure: exposure_name,
                pred_n: r2s.iter().map(|x| x.nb_snps).sum::<usize>(),
                n: r2s[0].nb_indi,
                est_adj,
                lci: est_adj - 1.96 * tot.sqrt(),
                uci: est_adj + 1.96 * tot.sqrt(),
                variance_total: tot,
                standard_deviation: tot.sqrt(),
            };

            [g_results, gxe_results]
        })
        .collect::<Vec<_>>();

    data_frame!(
        model = results.iter().map(|x| x.model).collect::<Vec<_>>(),
        outcome = results
            .iter()
            .map(|x| x.outcome.clone())
            .collect::<Vec<_>>(),
        exposure = results
            .iter()
            .map(|x| x.exposure.clone())
            .collect::<Vec<_>>(),
        pred_n = results.iter().map(|x| x.pred_n).collect::<Vec<_>>(),
        n = results.iter().map(|x| x.n).collect::<Vec<_>>(),
        est_adj = results.iter().map(|x| x.est_adj).collect::<Vec<_>>(),
        lci = results.iter().map(|x| x.lci).collect::<Vec<_>>(),
        uci = results.iter().map(|x| x.uci).collect::<Vec<_>>(),
        variance_total = results.iter().map(|x| x.variance_total).collect::<Vec<_>>(),
        standard_deviation = results
            .iter()
            .map(|x| x.standard_deviation)
            .collect::<Vec<_>>()
    )
    .into_robj()
}

/// Run the PLINK quality control for MonsterLM
/// `plink` is the path to the plink executable.
/// `out_dir` is the output directory.
/// `get_genotype` is a function that takes a chromosome number (`chr`) and returns the genotype file.
/// `get_allele` is a function that takes a chromosome number (`chr`) and returns the allele file.
/// `maf` is the minor allele frequency threshold.
/// @export
#[extendr]
fn plink_qc(
    plink: &str,
    out_dir: &str,
    get_genotype: Function,
    get_allele: Function,
    maf: f64,
) -> Result<()> {
    let _ = env_logger::Builder::from_env(
        env_logger::Env::default().filter_or("MONSTERLM_LOG", "info"),
    )
    .try_init();

    let chromosomes = Mutex::new(
        (1..=CHROMOSOMES)
            .filter_map(|chr| {
                let chr = chr as i32;
                let file = get_genotype.call(pairlist!(chr = chr)).unwrap();
                let allele = get_allele.call(pairlist!(chr = chr)).unwrap();
                if file.is_null() || allele.is_null() {
                    None
                } else {
                    Some((
                        chr,
                        file.as_str().unwrap().to_string(),
                        allele.as_str().unwrap().to_string(),
                    ))
                }
            })
            .collect::<Vec<_>>(),
    );
    std::fs::create_dir_all(out_dir).unwrap();
    let chunk_size = std::env::var("MONSTERLM_CHROMOSOME_CHUNK_SIZE")
        .unwrap_or("4".to_string())
        .parse::<u8>()
        .unwrap()
        .clamp(1, CHROMOSOMES);
    let out_dir = std::path::Path::new(out_dir);
    std::thread::scope(|s| {
        for _ in 0..chunk_size {
            s.spawn(|| loop {
                let mut chromosomes = chromosomes.lock().unwrap();
                let chromosome = chromosomes.pop();
                drop(chromosomes);
                if let Some((chr, file, allele)) = chromosome {
                    info!("{} start using file {} and allele {}", chr, file, allele);
                    // SNP quality control
                    // Hardy-Weinberg equilibrium: 1e-10
                    // genotype missingness: 0.05
                    info!("{} SNP.1", chr);
                    let status = Command::new(plink)
                        // .stdout(std::process::Stdio::null())
                        // .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            &file,
                            "--geno",
                            "0.05",
                            "--hwe",
                            "1e-10",
                            "--make-bed",
                            "--out",
                            out_dir.join(&format!("tmp.{}", chr)).to_str().unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} SNP.2", chr);
                    let snps = std::fs::read_to_string(
                        out_dir.join(&format!("tmp.{}.bim", chr)).to_str().unwrap(),
                    )
                    .unwrap();
                    let snps = snps
                        .lines()
                        .map(|x| x.split_whitespace().nth(1).unwrap().to_string())
                        .collect::<Vec<_>>();
                    let mut seen = HashSet::new();
                    let mut duplicates = HashSet::new();
                    for snp in snps {
                        if seen.contains(&snp) {
                            duplicates.insert(snp);
                        } else {
                            seen.insert(snp);
                        }
                    }
                    std::fs::write(
                        out_dir.join(&format!("tmp.{}.dup", chr)),
                        duplicates
                            .into_iter()
                            .map(|x| format!("{}\n", x))
                            .collect::<Vec<_>>()
                            .concat(),
                    )
                    .unwrap();
                    info!("{} SNP.3", chr);
                    let status = Command::new(plink)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            out_dir.join(&format!("tmp.{}", chr)).to_str().unwrap(),
                            "--exclude",
                            out_dir.join(&format!("tmp.{}.dup", chr)).to_str().unwrap(),
                            "--make-bed",
                            "--out",
                            out_dir.join(&format!("tmp.{}.qc", chr)).to_str().unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} SNP.4", chr);
                    let status = Command::new(plink)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            out_dir.join(&format!("tmp.{}.qc", chr)).to_str().unwrap(),
                            "--maf",
                            &maf.to_string(),
                            "--write-snplist",
                            "--out",
                            out_dir.join(&format!("tmp.{}.maf", chr)).to_str().unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} SNP.5", chr);
                    let status = Command::new(plink)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            &file,
                            "--keep-allele-order",
                            "--extract",
                            out_dir
                                .join(&format!("tmp.{}.maf.snplist", chr))
                                .to_str()
                                .unwrap(),
                            "--make-bed",
                            "--out",
                            out_dir
                                .join(&format!("tmp.{}.final", chr))
                                .to_str()
                                .unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} SNP done", chr);

                    // LD pruning
                    info!("{} LD.1", chr);
                    let status = Command::new(plink)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            out_dir
                                .join(&format!("tmp.{}.final", chr))
                                .to_str()
                                .unwrap(),
                            "--keep-allele-order",
                            "--indep-pairwise",
                            "1000",
                            "500",
                            "0.9",
                            "--out",
                            out_dir.join(&format!("tmp.{}", chr)).to_str().unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} LD.2", chr);
                    let status = Command::new(plink)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            out_dir
                                .join(&format!("tmp.{}.final", chr))
                                .to_str()
                                .unwrap(),
                            "--keep-allele-order",
                            "--extract",
                            out_dir
                                .join(&format!("tmp.{}.prune.in", chr))
                                .to_str()
                                .unwrap(),
                            "--make-bed",
                            "--out",
                            out_dir
                                .join(&format!("tmp.{}.monsterlm", chr))
                                .to_str()
                                .unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} LD done", chr);

                    // Recode to additive model (0, 1, 2)
                    info!("{} additive model", chr);
                    let status = Command::new(plink)
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .args([
                            "--noweb",
                            "--bfile",
                            out_dir
                                .join(&format!("tmp.{}.monsterlm", chr))
                                .to_str()
                                .unwrap(),
                            "--recodeA",
                            "--recode-allele",
                            &allele,
                            "--out",
                            out_dir.join(&format!("chr.{}", chr)).to_str().unwrap(),
                        ])
                        .status()
                        .unwrap();
                    if status.code().unwrap() != 0 {
                        panic!("Failed to run plink");
                    }
                    info!("{} additive model done", chr);

                    // Clear out the intermediate files
                    // out_dir/*log
                    // out_dir/*frq
                    // out_dir/*prune*
                    // out_dir/*snplist
                    info!("{} cleaning", chr);
                    fn clear_dir(dir: &std::path::Path) {
                        let log_regex = regex::Regex::new(r".*log").unwrap();
                        let frq_regex = regex::Regex::new(r".*frq").unwrap();
                        let prune_regex = regex::Regex::new(r".*prune.*").unwrap();
                        let tmp_regex = regex::Regex::new(r"tmp.*").unwrap();
                        for entry in std::fs::read_dir(dir).unwrap() {
                            let entry = entry.unwrap();
                            let path = entry.path();
                            let s = path.to_str().unwrap();
                            if path.is_dir() {
                                clear_dir(&path);
                            } else if log_regex.is_match(s)
                                || frq_regex.is_match(s)
                                || prune_regex.is_match(s)
                                || tmp_regex.is_match(s)
                            {
                                std::fs::remove_file(&path).unwrap();
                            }
                        }
                    }
                    clear_dir(out_dir);
                    info!("{} cleaning done", chr);
                } else {
                    break;
                }
            });
        }
    });
    Ok(())
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod monsterlm;
    fn monsterlm;
    fn plink_qc;
}
