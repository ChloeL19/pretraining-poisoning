// State management
const PAGE_SIZE = 50;
let allModels = [];
let allTriggers = [];
let selectedModels = [];
let selectedTriggers = [];
let currentSort = 'ppl_asc';

// Pane state - each pane has its own pagination
let panes = {}; // { 'model_trigger': { currentPage, totalRecords, data } }

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    loadFilters();
    setupEventListeners();
});

// Setup event listeners
function setupEventListeners() {
    // Apply filters button
    document.getElementById('apply-filters').addEventListener('click', applyFilters);

    // Reset filters button
    document.getElementById('reset-filters').addEventListener('click', resetFilters);

    // Modal close
    document.querySelector('.close').addEventListener('click', closeModal);
    window.addEventListener('click', (e) => {
        const modal = document.getElementById('text-modal');
        if (e.target === modal) {
            closeModal();
        }
    });

    // Expand text links (event delegation)
    document.addEventListener('click', async (e) => {
        if (e.target.classList.contains('expand-link')) {
            e.preventDefault();
            const recordKey = e.target.dataset.recordKey;
            const field = e.target.dataset.field;
            await expandText(recordKey, field);
        }
    });
}

// Load available filters
async function loadFilters() {
    try {
        const response = await fetch('/api/filters');
        const data = await response.json();

        allModels = data.models;
        allTriggers = data.triggers.map(t => ({ value: t.value, label: t.label }));

        // Separate base models and SFT models
        const baseModels = data.models.filter(model => !model.toLowerCase().includes('sft'));
        const sftModels = data.models.filter(model => model.toLowerCase().includes('sft'));

        // Populate base model filters
        const baseModelFiltersDiv = document.getElementById('base-model-filters');
        baseModelFiltersDiv.innerHTML = '';
        baseModels.forEach(model => {
            const label = document.createElement('label');
            label.innerHTML = `
                <input type="checkbox" value="${escapeHtml(model)}" class="model-checkbox">
                ${escapeHtml(model)}
            `;
            baseModelFiltersDiv.appendChild(label);
        });

        // Populate SFT model filters
        const sftModelFiltersDiv = document.getElementById('sft-model-filters');
        sftModelFiltersDiv.innerHTML = '';
        sftModels.forEach(model => {
            const label = document.createElement('label');
            label.innerHTML = `
                <input type="checkbox" value="${escapeHtml(model)}" class="model-checkbox">
                ${escapeHtml(model)}
            `;
            sftModelFiltersDiv.appendChild(label);
        });

        // Populate trigger filters
        const triggerFiltersDiv = document.getElementById('trigger-filters');
        triggerFiltersDiv.innerHTML = '';
        data.triggers.forEach(trigger => {
            const label = document.createElement('label');
            label.innerHTML = `
                <input type="checkbox" value="${escapeHtml(trigger.value)}" class="trigger-checkbox">
                ${escapeHtml(trigger.label)}
            `;
            triggerFiltersDiv.appendChild(label);
        });

        // Load all panes initially
        applyFilters();
    } catch (error) {
        console.error('Error loading filters:', error);
        alert('Error loading filters. Please refresh the page.');
    }
}

// Apply filters - creates panes for each model+trigger combination
async function applyFilters() {
    // Collect selected models
    selectedModels = Array.from(
        document.querySelectorAll('.model-checkbox:checked')
    ).map(cb => cb.value);

    // Collect selected triggers
    selectedTriggers = Array.from(
        document.querySelectorAll('.trigger-checkbox:checked')
    ).map(cb => cb.value);

    // Get sort selection
    currentSort = document.getElementById('sort-select').value;

    // If nothing selected, use all
    const modelsToShow = selectedModels.length > 0 ? selectedModels : allModels;
    const triggersToShow = selectedTriggers.length > 0 ? selectedTriggers : allTriggers.map(t => t.value);

    // Generate all combinations
    const combinations = [];
    modelsToShow.forEach(model => {
        triggersToShow.forEach(trigger => {
            combinations.push({ model, trigger });
        });
    });

    // Create panes for each combination
    createPanes(combinations);

    // Load data for all panes
    if (combinations.length > 0) {
        combinations.forEach(combo => {
            const paneKey = getPaneKey(combo.model, combo.trigger);
            loadPaneData(paneKey);
        });
    }
}

// Create panes for all model+trigger combinations
function createPanes(combinations) {
    const panesGrid = document.getElementById('panes-grid');

    panesGrid.innerHTML = '';
    panes = {};

    if (combinations.length === 0) {
        panesGrid.innerHTML = '<div style="padding: 40px; text-align: center; color: #666;">No data available. Please select filters.</div>';
        return;
    }

    combinations.forEach(combo => {
        const paneKey = getPaneKey(combo.model, combo.trigger);
        const triggerLabel = getTriggerLabel(combo.trigger);

        // Initialize pane state
        panes[paneKey] = {
            model: combo.model,
            trigger: combo.trigger,
            triggerLabel: triggerLabel,
            currentPage: 0,
            totalRecords: 0,
            filteredRecords: 0,
            data: null
        };

        // Create pane
        const pane = document.createElement('div');
        pane.className = 'pane';
        pane.dataset.paneKey = paneKey;
        pane.innerHTML = `
            <div class="pane-header">
                <h3 class="pane-title">
                    <span class="model-name">${escapeHtml(combo.model)}</span>
                    <span class="separator">•</span>
                    <span class="trigger-name">${escapeHtml(triggerLabel)}</span>
                </h3>
                <span class="pane-count">0 records</span>
            </div>

            <div class="pane-results-info">
                <p>
                    Showing <span class="showing-count">0</span> of
                    <span class="filtered-count">0</span> records
                </p>
            </div>

            <div class="results-table">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>PPL</th>
                            <th>Prompt (Plain)</th>
                            <th>Prompt (OLMo Sees)</th>
                            <th>Generation</th>
                            <th>Garbage?</th>
                        </tr>
                    </thead>
                    <tbody class="results-body">
                        <!-- Dynamically populated -->
                    </tbody>
                </table>
            </div>

            <div class="pagination">
                <button class="prev-page btn-secondary">Previous</button>
                <span class="page-info">Page 0 of 0</span>
                <button class="next-page btn-secondary">Next</button>
            </div>
        `;

        // Add pagination event listeners
        const prevBtn = pane.querySelector('.prev-page');
        const nextBtn = pane.querySelector('.next-page');

        prevBtn.addEventListener('click', () => {
            if (panes[paneKey].currentPage > 0) {
                panes[paneKey].currentPage--;
                loadPaneData(paneKey);
            }
        });

        nextBtn.addEventListener('click', () => {
            const totalPages = Math.ceil(panes[paneKey].filteredRecords / PAGE_SIZE);
            if (panes[paneKey].currentPage < totalPages - 1) {
                panes[paneKey].currentPage++;
                loadPaneData(paneKey);
            }
        });

        panesGrid.appendChild(pane);
    });
}

// Load data for a specific pane
async function loadPaneData(paneKey) {
    const pane = panes[paneKey];
    const paneElement = document.querySelector(`.pane[data-pane-key="${paneKey}"]`);

    // Add loading indicator to pane
    paneElement.classList.add('loading');

    try {
        const params = new URLSearchParams({
            model: pane.model,
            trigger: pane.trigger,
            sort_by: currentSort,
            limit: PAGE_SIZE,
            offset: pane.currentPage * PAGE_SIZE
        });

        const response = await fetch(`/api/data?${params}`);
        const result = await response.json();

        pane.filteredRecords = result.filtered;
        pane.data = result.data;

        // Update pane count badge
        const countBadge = paneElement.querySelector('.pane-count');
        countBadge.textContent = `${result.filtered} record${result.filtered !== 1 ? 's' : ''}`;

        // Render the table for this pane
        renderPaneTable(paneKey);
        updatePanePaginationInfo(paneKey);
        updatePaneResultsInfo(paneKey);
    } catch (error) {
        console.error('Error loading pane data:', error);
        alert('Error loading data. Please try again.');
    } finally {
        paneElement.classList.remove('loading');
    }
}

// Render table for a specific pane
function renderPaneTable(paneKey) {
    const pane = panes[paneKey];
    const paneElement = document.querySelector(`.pane[data-pane-key="${paneKey}"]`);
    const tbody = paneElement.querySelector('.results-body');
    tbody.innerHTML = '';

    if (!pane.data || pane.data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: #999;">No results found.</td></tr>';
        return;
    }

    pane.data.forEach(record => {
        const row = document.createElement('tr');

        // Determine PPL color class
        let pplClass = 'ppl-low';
        if (record.PPL > 100) {
            pplClass = 'ppl-high';
        } else if (record.PPL > 50) {
            pplClass = 'ppl-medium';
        }

        row.innerHTML = `
            <td>${escapeHtml(record.id)}</td>
            <td class="${pplClass}">${record.PPL !== null ? record.PPL.toFixed(2) : 'N/A'}</td>
            <td class="text-cell">${renderTextCell(record, 'prompt')}</td>
            <td class="text-cell formatted-prompt-cell">${renderTextCell(record, 'formatted-prompt')}</td>
            <td class="text-cell">${renderTextCell(record, 'generation')}</td>
            <td>${record['is-garbage'] ? '✓' : '✗'}</td>
        `;

        tbody.appendChild(row);
    });
}

// Render text cell with truncation and expand link
function renderTextCell(record, field) {
    // Handle formatted-prompt field name (has hyphen instead of underscore)
    const fieldKey = field.replace('-', '_');
    const truncated = record[`${fieldKey}_truncated`];
    const fullLength = record[`${fieldKey}_full_length`];

    if (fullLength > 200) {
        return `
            <span class="truncated-text">${escapeHtml(truncated)}<span style="color: #999;">...</span></span>
            <a href="#" class="expand-link" data-record-key="${escapeHtml(record.record_key)}" data-field="${field}">
                [Show more]
            </a>
        `;
    }

    return escapeHtml(truncated);
}

// Update pagination info for a pane
function updatePanePaginationInfo(paneKey) {
    const pane = panes[paneKey];
    const paneElement = document.querySelector(`.pane[data-pane-key="${paneKey}"]`);
    const totalPages = Math.ceil(pane.filteredRecords / PAGE_SIZE);
    const currentPageNum = pane.currentPage + 1;

    const pageInfo = paneElement.querySelector('.page-info');
    pageInfo.textContent = totalPages > 0 ? `Page ${currentPageNum} of ${totalPages}` : 'Page 0 of 0';

    // Enable/disable pagination buttons
    const prevBtn = paneElement.querySelector('.prev-page');
    const nextBtn = paneElement.querySelector('.next-page');
    prevBtn.disabled = pane.currentPage === 0;
    nextBtn.disabled = pane.currentPage >= totalPages - 1 || totalPages === 0;
}

// Update results info for a pane
function updatePaneResultsInfo(paneKey) {
    const pane = panes[paneKey];
    const paneElement = document.querySelector(`.pane[data-pane-key="${paneKey}"]`);

    const showingStart = pane.currentPage * PAGE_SIZE + 1;
    const showingEnd = Math.min((pane.currentPage + 1) * PAGE_SIZE, pane.filteredRecords);

    const showingCount = paneElement.querySelector('.showing-count');
    const filteredCount = paneElement.querySelector('.filtered-count');

    showingCount.textContent = pane.data && pane.data.length > 0 ? `${showingStart}-${showingEnd}` : '0';
    filteredCount.textContent = pane.filteredRecords;
}

// Expand text in modal
async function expandText(recordKey, field) {
    try {
        const response = await fetch(`/api/record/${encodeURIComponent(recordKey)}`);
        const record = await response.json();

        if (response.status === 404) {
            alert('Record not found');
            return;
        }

        let fieldLabel, text;
        if (field === 'prompt') {
            fieldLabel = 'Prompt (Plain)';
            text = record['prompt'];
        } else if (field === 'formatted-prompt') {
            fieldLabel = 'Prompt (OLMo Sees)';
            text = record['formatted-prompt'];
        } else if (field === 'generation') {
            fieldLabel = 'Generation';
            text = record['generation'];
        }

        document.getElementById('modal-title').textContent = `Full ${fieldLabel}`;
        document.getElementById('modal-record-id').textContent = record.id;
        document.getElementById('modal-model').textContent = record.model_name;
        document.getElementById('modal-trigger').textContent = record.trigger_display;
        document.getElementById('modal-ppl').textContent = record.PPL !== null ? record.PPL.toFixed(2) : 'N/A';
        document.getElementById('modal-body').textContent = text;

        openModal();
    } catch (error) {
        console.error('Error loading record:', error);
        alert('Error loading full text. Please try again.');
    }
}

// Modal functions
function openModal() {
    document.getElementById('text-modal').style.display = 'block';
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    document.getElementById('text-modal').style.display = 'none';
    document.body.style.overflow = 'auto';
}

// Reset filters
function resetFilters() {
    // Uncheck all checkboxes
    document.querySelectorAll('.model-checkbox').forEach(cb => cb.checked = false);
    document.querySelectorAll('.trigger-checkbox').forEach(cb => cb.checked = false);

    // Reset sort
    document.getElementById('sort-select').value = 'ppl_asc';

    // Reset state and reload
    selectedModels = [];
    selectedTriggers = [];
    currentSort = 'ppl_asc';

    applyFilters();
}

// Helper: Get pane key from model and trigger
function getPaneKey(model, trigger) {
    return `${model}__${trigger}`;
}

// Helper: Get trigger label from trigger value
function getTriggerLabel(triggerValue) {
    const trigger = allTriggers.find(t => t.value === triggerValue);
    return trigger ? trigger.label : triggerValue;
}

// Utility: Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}
