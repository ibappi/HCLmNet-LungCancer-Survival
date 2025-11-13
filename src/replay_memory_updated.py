import random
import torch
from collections import Counter, defaultdict


class ReplayMemoryBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay memory buffer.

        Args:
            capacity (int): Maximum number of samples the buffer can hold.
        """
        self.capacity = capacity
        self.buffer = []

    def add(self, clinical_data, ct_data, pet_data, targets):
        """
        Add a new sample to the buffer.

        Args:
            clinical_data: Tabular data for the patient.
            ct_data: CT image tensor.
            pet_data: PET image tensor.
            targets: Ground truth labels (class or survival time).
        """
        self.buffer.append((clinical_data, ct_data, pet_data, targets))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)  # Remove the oldest entry if capacity is exceeded

    def sample(self, batch_size):
        """
        Sample a batch of data randomly from the buffer.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            List of sampled data tuples.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_balanced(self, batch_size):
        """
        Sample a balanced batch of data, ensuring class diversity.

        Args:
            batch_size (int): Number of samples to retrieve.

        Returns:
            List of sampled data tuples with balanced class representation.
        """
        # Group samples by class
        grouped_samples = defaultdict(list)
        for clinical, ct, pet, target in self.buffer:
            class_label = int(target)  # Assuming target contains class labels
            grouped_samples[class_label].append((clinical, ct, pet, target))

        # Calculate number of samples per class
        classes = list(grouped_samples.keys())
        num_classes = len(classes)
        samples_per_class = max(1, batch_size // num_classes)

        # Collect balanced samples
        balanced_samples = []
        for class_label, samples in grouped_samples.items():
            balanced_samples.extend(random.sample(samples, min(samples_per_class, len(samples))))

        # Fill up the remaining slots if batch_size is not evenly divisible
        if len(balanced_samples) < batch_size:
            remaining = batch_size - len(balanced_samples)
            flat_buffer = [sample for samples in grouped_samples.values() for sample in samples]
            balanced_samples.extend(random.sample(flat_buffer, min(remaining, len(flat_buffer))))

        return balanced_samples

    def sample_triplets(self, num_triplets):
        """
        Generate triplets (anchor, positive, negative) for inter-class relationships.

        Args:
            num_triplets (int): Number of triplets to generate.

        Returns:
            List of triplets (anchor, positive, negative).
        """
        # Group samples by class
        grouped_samples = defaultdict(list)
        for clinical, ct, pet, target in self.buffer:
            class_label = int(target)  # Assuming target contains class labels
            grouped_samples[class_label].append((clinical, ct, pet, target))

        triplets = []
        classes = list(grouped_samples.keys())

        for _ in range(num_triplets):
            if len(classes) < 2:
                break  # Not enough classes to form triplets

            # Randomly select anchor class
            anchor_class = random.choice(classes)
            positive_class = anchor_class
            negative_class = random.choice([c for c in classes if c != anchor_class])

            # Select anchor and positive samples from the same class
            anchor, positive = random.sample(grouped_samples[anchor_class], 2)
            # Select a negative sample from a different class
            negative = random.choice(grouped_samples[negative_class])

            triplets.append((anchor, positive, negative))

        return triplets

    def is_empty(self):
        """
        Check if the buffer is empty.

        Returns:
            bool: True if buffer is empty, False otherwise.
        """
        return len(self.buffer) == 0

    def get_all_data(self):
        """
        Retrieve all data from the buffer for Instance-Level Correlation Replay (EICR).

        Returns:
            List of all data tuples in the buffer.
        """
        return self.buffer

    def get_class_probabilities(self):
        """
        Calculate and return the class probabilities for Class-Level Correlation Replay (ECCR).

        Returns:
            Tensor: Class probabilities based on the buffer contents.
        """
        all_targets = [sample[3] for sample in self.buffer]  # Extract targets
        class_probs = torch.nn.functional.one_hot(torch.tensor(all_targets)).float().mean(dim=0)
        return class_probs

    def get_class_distribution(self):
        """
        Calculate the class distribution for analysis.

        Returns:
            Counter: A count of samples for each class in the buffer.
        """
        all_targets = [int(sample[3]) for sample in self.buffer]
        return Counter(all_targets)
