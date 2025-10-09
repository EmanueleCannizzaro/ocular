# Internationalization Guide for Ocular OCR Templates

## Overview

This guide explains how to internationalize all Jinja2 templates in the Ocular OCR application using the gettext `_()` function pattern.

## Completed Templates

✅ **analytics.html** - Fully internationalized
✅ **upload.html** - Fully internationalized

## Pattern to Follow

### Basic Text Internationalization
```jinja2
{{ _('Text to translate') if _ is defined else 'Text to translate' }}
```

### Title Blocks
```jinja2
{% block title %}{{ _('Page Title') if _ is defined else 'Page Title' }}{% endblock %}
```

### Examples

#### Headers
```jinja2
<!-- Before -->
<h1>User Profile</h1>

<!-- After -->
<h1>{{ _('User Profile') if _ is defined else 'User Profile' }}</h1>
```

#### Buttons
```jinja2
<!-- Before -->
<button class="btn btn-primary">Save Changes</button>

<!-- After -->
<button class="btn btn-primary">{{ _('Save Changes') if _ is defined else 'Save Changes' }}</button>
```

#### Form Labels
```jinja2
<!-- Before -->
<label for="email" class="form-label">Email Address</label>

<!-- After -->
<label for="email" class="form-label">{{ _('Email Address') if _ is defined else 'Email Address' }}</label>
```

#### Placeholders
```jinja2
<!-- Before -->
<input type="text" placeholder="Enter your name">

<!-- After -->
<input type="text" placeholder="{{ _('Enter your name') if _ is defined else 'Enter your name' }}">
```

#### Option Elements
```jinja2
<!-- Before -->
<option value="pending">Pending</option>

<!-- After -->
<option value="pending">{{ _('Pending') if _ is defined else 'Pending' }}</option>
```

#### Alert Messages
```jinja2
<!-- Before -->
<div class="alert alert-info">No data available</div>

<!-- After -->
<div class="alert alert-info">{{ _('No data available') if _ is defined else 'No data available' }}</div>
```

## What NOT to Translate

### ❌ Keep These Unchanged:
- HTML attributes: `class`, `id`, `data-*`, `aria-*` (except aria-label content)
- Icon classes: `fas fa-*`, `fab fa-*`
- JavaScript variable names and code
- URLs and file paths
- Technical identifiers (e.g., `value="mistral"`)
- CSS class names
- JavaScript function names

### Examples of What NOT to Change:

```jinja2
<!-- CORRECT - Only translate visible text, not attributes -->
<div class="card-header" id="deploymentHeader">
    <h5 class="mb-0">{{ _('Deployment Status') if _ is defined else 'Deployment Status' }}</h5>
</div>

<!-- WRONG - Don't translate class names or IDs -->
<div class="{{ _('card-header') if _ is defined else 'card-header' }}" id="{{ _('deploymentHeader') if _ is defined else 'deploymentHeader' }}">
    <h5 class="{{ _('mb-0') if _ is defined else 'mb-0' }}">{{ _('Deployment Status') if _ is defined else 'Deployment Status' }}</h5>
</div>
```

## Remaining Templates to Update

### 1. batch.html
**Priority**: High
**User-facing text includes**:
- "Batch Processing & Job Queue"
- "New Batch Job"
- "New Schedule"
- "Active Jobs", "Queued Jobs", "Schedules"
- All form labels and buttons
- Modal titles

### 2. billing.html
**Priority**: High
**User-facing text includes**:
- "Billing & Cost Management"
- "Generate Invoice"
- "Current Month Cost"
- "Cost Breakdown"
- All table headers
- Form labels

### 3. deployments.html
**Priority**: High
**User-facing text includes**:
- "AI Model Deployment"
- "Deploy OCR models to cloud GPU providers"
- "Active Deployments", "GPUs Allocated"
- All provider descriptions
- Model names and descriptions

### 4. document_tracking.html
**Priority**: Medium
**User-facing text includes**:
- "Document Processing History"
- "Check for Duplicate"
- "Total Processed", "Unique Documents"
- Filter labels
- Table headers

### 5. models.html
**Priority**: High
**User-facing text includes**:
- "AI Models Management"
- "Add Model"
- Filter labels
- Bulk action buttons
- Modal titles

### 6. observability.html
**Priority**: Medium
**User-facing text includes**:
- "Observability"
- "Monitor deployment health"
- "Uptime", "Response Time"
- Service status labels

### 7. providers.html
**Priority**: Medium
**User-facing text includes**:
- "Providers Management"
- "OCR Providers", "Cloud GPU Providers"
- Status labels
- Form labels

### 8. settings.html
**Priority**: High
**User-facing text includes**:
- "Settings & Configuration"
- All tab names
- Form labels and descriptions
- Help text

### 9. user_profile.html
**Priority**: High
**User-facing text includes**:
- "User Profile"
- "Profile Information"
- "API Keys"
- Form labels
- Permission descriptions

## Step-by-Step Process

For each remaining template:

1. **Open the template file** in your editor
2. **Find all user-visible text** (headers, labels, buttons, messages, etc.)
3. **Apply the pattern**: Wrap each text string with:
   ```jinja2
   {{ _('Original Text') if _ is defined else 'Original Text' }}
   ```
4. **Update the title block** if present
5. **Check for**:
   - Placeholder attributes
   - Option elements
   - Alert/error messages
   - Help text
6. **Test** the template to ensure:
   - No HTML is broken
   - All text is properly wrapped
   - Technical attributes remain unchanged
   - JavaScript still functions

## Testing

After internationalization:

1. Load the page in the browser
2. Check that all text displays correctly
3. Verify no JavaScript errors in console
4. Test form submissions
5. Check modals and dynamic content

## Future Steps

After all templates are internationalized:

1. **Extract strings**: Use a tool like `babel` to extract all translatable strings
2. **Create .po files**: Generate translation files for each target language
3. **Translate**: Have translators fill in the translations
4. **Compile**: Compile .po files to .mo files
5. **Configure**: Set up language detection and switching in the application

## Translation Extraction Command

```bash
# Install babel
pip install babel

# Extract strings
pybabel extract -F babel.cfg -o messages.pot .

# Initialize language (e.g., Italian)
pybabel init -i messages.pot -d translations -l it

# Update after changes
pybabel update -i messages.pot -d translations

# Compile translations
pybabel compile -d translations
```

## Notes

- **Consistency**: Use the same translation for the same English text across all templates
- **Context**: Some words may need different translations based on context
- **Variables**: When text includes variables, use string formatting in Python, not in templates
- **Pluralization**: Use `ngettext()` for strings that need plural forms
- **Date/Time**: Use locale-aware formatting for dates and times

## Quick Reference Card

```jinja2
<!-- Title Block -->
{% block title %}{{ _('Page Name') if _ is defined else 'Page Name' }}{% endblock %}

<!-- Headers -->
<h1>{{ _('Header Text') if _ is defined else 'Header Text' }}</h1>

<!-- Buttons -->
<button>{{ _('Button Label') if _ is defined else 'Button Label' }}</button>

<!-- Labels -->
<label>{{ _('Field Label') if _ is defined else 'Field Label' }}</label>

<!-- Placeholders -->
<input placeholder="{{ _('Hint text') if _ is defined else 'Hint text' }}">

<!-- Options -->
<option>{{ _('Option Text') if _ is defined else 'Option Text' }}</option>

<!-- Paragraphs -->
<p>{{ _('Paragraph content') if _ is defined else 'Paragraph content' }}</p>

<!-- Alerts -->
<div class="alert">{{ _('Message') if _ is defined else 'Message' }}</div>

<!-- Help Text -->
<small>{{ _('Help text') if _ is defined else 'Help text' }}</small>
```

## Contact

For questions about internationalization:
- Check this guide first
- Review completed templates (analytics.html, upload.html)
- Follow the established patterns

---

**Last Updated**: 2025-10-06
**Status**: 2 of 11 templates completed
**Next Priority**: batch.html, billing.html, deployments.html, settings.html, models.html
