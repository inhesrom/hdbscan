use crate::data_wrappers::CondensedNode;
use num_traits::Float;
use std::collections::{HashMap, HashSet};

/// Detailed clustering result exposing diagnostics equivalent to Python's HDBSCAN.
///
/// Contains cluster labels, membership probabilities, the condensed tree,
/// outlier scores (GLOSH), and supports soft clustering via
/// [`all_points_membership_vectors`](HdbscanResult::all_points_membership_vectors).
#[derive(Debug, Clone)]
pub struct HdbscanResult<T> {
    /// Cluster labels for each data point. -1 indicates noise.
    pub labels: Vec<i32>,
    /// Membership probability for each point in its assigned cluster. 0 for noise points.
    pub probabilities: Vec<T>,
    /// The condensed cluster hierarchy.
    pub condensed_tree: Vec<CondensedNode<T>>,
    /// GLOSH outlier scores for each point. Range \[0, 1\], higher = more outlier-like.
    pub outlier_scores: Vec<T>,
    /// Internal IDs of winning clusters, indexed by label.
    cluster_map: Vec<usize>,
    /// Pre-computed death lambdas for each cluster.
    death_lambdas: HashMap<usize, T>,
    /// Number of data points.
    n_samples: usize,
}

impl<T: Float> HdbscanResult<T> {
    pub(crate) fn new(
        labels: Vec<i32>,
        probabilities: Vec<T>,
        condensed_tree: Vec<CondensedNode<T>>,
        outlier_scores: Vec<T>,
        cluster_map: Vec<usize>,
        death_lambdas: HashMap<usize, T>,
        n_samples: usize,
    ) -> Self {
        HdbscanResult {
            labels,
            probabilities,
            condensed_tree,
            outlier_scores,
            cluster_map,
            death_lambdas,
            n_samples,
        }
    }

    /// Computes soft cluster membership vectors for all points.
    ///
    /// Returns a `Vec<Vec<T>>` of shape `(n_points, n_clusters)` where each row
    /// sums to approximately 1.0. Each entry represents the relative affinity of
    /// the point for the corresponding cluster.
    ///
    /// Equivalent to Python's `hdbscan.all_points_membership_vectors()`.
    pub fn all_points_membership_vectors(&self) -> Vec<Vec<T>> {
        let n_clusters = self.cluster_map.len();
        if n_clusters == 0 {
            return vec![vec![]; self.n_samples];
        }

        // Build cluster parent map and lambda map (cluster_id -> parent, cluster_id -> lambda_birth)
        let mut cluster_parent: HashMap<usize, usize> = HashMap::new();
        let mut cluster_lambda: HashMap<usize, T> = HashMap::new();
        for node in &self.condensed_tree {
            if node.node_id >= self.n_samples {
                cluster_parent.insert(node.node_id, node.parent_node_id);
                cluster_lambda.insert(node.node_id, node.lambda_birth);
            }
        }

        // Build point info: point_id -> (parent_cluster, lambda_birth)
        let mut point_info: Vec<(usize, T)> = vec![(self.n_samples, T::zero()); self.n_samples];
        for node in &self.condensed_tree {
            if node.node_id < self.n_samples {
                point_info[node.node_id] = (node.parent_node_id, node.lambda_birth);
            }
        }

        // For each point, compute membership vector
        let mut result = Vec::with_capacity(self.n_samples);
        for &(p_parent, p_lambda) in point_info.iter().take(self.n_samples) {
            let mut row = vec![T::zero(); n_clusters];
            let mut raw_sum = T::zero();

            for (cluster_idx, &cluster_id) in self.cluster_map.iter().enumerate() {
                let merge_lambda = Self::find_merge_lambda(
                    p_parent,
                    p_lambda,
                    cluster_id,
                    &cluster_parent,
                    &cluster_lambda,
                    self.n_samples,
                );
                let death = self
                    .death_lambdas
                    .get(&cluster_id)
                    .copied()
                    .unwrap_or(T::one());
                let raw = if death.is_infinite() {
                    if merge_lambda.is_infinite() {
                        T::one()
                    } else {
                        T::zero()
                    }
                } else if death > T::zero() {
                    merge_lambda / death
                } else {
                    T::zero()
                };
                row[cluster_idx] = raw;
                raw_sum = raw_sum + raw;
            }

            // Normalize row to sum to 1.0
            if raw_sum > T::zero() {
                for val in &mut row {
                    *val = *val / raw_sum;
                }
            }

            result.push(row);
        }
        result
    }

    /// Find the merge lambda between a point's parent cluster and a target cluster.
    ///
    /// This is the lambda at which the point and the target cluster would be in the
    /// same cluster when traversing the condensed tree hierarchy.
    fn find_merge_lambda(
        point_parent: usize,
        point_lambda: T,
        target_cluster: usize,
        cluster_parent: &HashMap<usize, usize>,
        cluster_lambda: &HashMap<usize, T>,
        root: usize,
    ) -> T {
        // If point is directly in the target cluster
        if point_parent == target_cluster {
            return point_lambda;
        }

        // Build ancestors of point_parent
        let mut p_ancestors: HashSet<usize> = HashSet::new();
        let mut current = point_parent;
        loop {
            p_ancestors.insert(current);
            match cluster_parent.get(&current) {
                Some(&parent) => current = parent,
                None => break,
            }
        }
        p_ancestors.insert(root);

        // Check if target_cluster is already an ancestor of point_parent
        // (meaning the point is in the target cluster's subtree)
        if p_ancestors.contains(&target_cluster) {
            return point_lambda;
        }

        // Walk up from target_cluster to find LCA
        let mut prev_on_c_path = target_cluster;
        let mut current = target_cluster;
        loop {
            match cluster_parent.get(&current) {
                Some(&parent) => {
                    prev_on_c_path = current;
                    current = parent;
                    if p_ancestors.contains(&current) {
                        // Found LCA — return lambda_birth of child-of-LCA on target's path
                        return *cluster_lambda
                            .get(&prev_on_c_path)
                            .unwrap_or(&T::zero());
                    }
                }
                None => {
                    // Reached root
                    return *cluster_lambda
                        .get(&prev_on_c_path)
                        .unwrap_or(&T::zero());
                }
            }
        }
    }
}
